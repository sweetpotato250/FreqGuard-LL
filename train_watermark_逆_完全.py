import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import argparse
from tqdm import tqdm
import numpy as np

# 导入 EndoGS 核心模块
from gaussian_core.provider import EndoDataset
from gaussian_core.utils import seed_everything
from gaussian_core.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss
from utils.image_utils import psnr


# ==============================================================================
# SECTION 1: INN Core Classes
# ==============================================================================

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return torch.cat([x_LL, x_HL, x_LH, x_HH], dim=1)


class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        in_batch, in_channel, in_height, in_width = x.size()
        out_channel = int(in_channel // 4)
        out_height = 2 * in_height
        out_width = 2 * in_width
        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
        h = torch.zeros([in_batch, out_channel, out_height, out_width]).float().to(x.device)
        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
        return h


class CouplingLayer(nn.Module):
    def __init__(self, in_channels, mid_channels=64):
        super(CouplingLayer, self).__init__()
        self.split_len1 = in_channels // 2
        self.split_len2 = in_channels - self.split_len1
        self.net = nn.Sequential(
            nn.Conv2d(self.split_len1, mid_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(mid_channels, self.split_len2 * 2, 3, 1, 1)
        )

    def forward(self, x, reverse=False):
        x1, x2 = torch.split(x, [self.split_len1, self.split_len2], dim=1)
        st = self.net(x1)
        s, t = torch.chunk(st, 2, dim=1)
        s = torch.sigmoid(s + 2) + 1e-6
        if not reverse:
            return torch.cat([x1, s * x2 + t], dim=1)
        else:
            return torch.cat([x1, (x2 - t) / s], dim=1)


class WatermarkINN(nn.Module):
    def __init__(self, in_channels=12, steps=4, wm_len=64, alpha=0.1):
        super(WatermarkINN, self).__init__()
        self.dwt = DWT();
        self.idwt = IDWT()
        self.wm_len = wm_len;
        self.alpha = alpha
        self.layers = nn.ModuleList([CouplingLayer(in_channels) for _ in range(steps)])
        self.wm_projector = nn.Sequential(nn.Linear(wm_len, in_channels), nn.ReLU(),
                                          nn.Linear(in_channels, in_channels))
        self.wm_extractor = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(in_channels, 128),
                                          nn.ReLU(), nn.Linear(128, wm_len))

    def get_wm_feature(self, watermark, dims):
        return self.wm_projector(watermark).unsqueeze(-1).unsqueeze(-1).expand(*dims)

    def inn_forward(self, x):
        for layer in self.layers: x = layer(x, reverse=False)
        return x

    def inn_inverse(self, z):
        for layer in reversed(self.layers): z = layer(z, reverse=True)
        return z

    def embed(self, image, watermark):
        coeffs = self.dwt(image)
        z = self.inn_forward(coeffs)
        wm_feat = self.get_wm_feature(watermark, z.shape)
        z_watermarked = z + self.alpha * wm_feat
        return self.idwt(self.inn_inverse(z_watermarked))

    def extract(self, watermarked_image, gt_watermark=None):
        coeffs_wm = self.dwt(watermarked_image)
        z_rec = self.inn_forward(coeffs_wm)
        w_logits = self.wm_extractor(z_rec)
        w_to_subtract = gt_watermark if gt_watermark is not None else (torch.sigmoid(w_logits) > 0.5).float()
        z_clean = z_rec - self.alpha * self.get_wm_feature(w_to_subtract, z_rec.shape)
        return self.idwt(self.inn_inverse(z_clean)), w_logits


def compute_ber_loss(w_pred_logits, w_gt):
    return F.binary_cross_entropy_with_logits(w_pred_logits, w_gt)


def compute_accuracy(w_pred_logits, w_gt):
    return ((torch.sigmoid(w_pred_logits) > 0.5).float() == w_gt).float().mean().item()


# ==============================================================================
# SECTION 2: Training Logic
# ==============================================================================

def warmup_inn(inn_model, dataloader, watermark_key, fixed_gaussians, epochs=20):
    print(f"\n[Phase 1] Warming up INN (Target: Pretrained Render)...")
    optimizer = optim.Adam(inn_model.parameters(), lr=1e-3)
    inn_model.train()
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    for epoch in range(epochs):
        epoch_acc = 0
        limit_batches = 50
        pbar = tqdm(enumerate(dataloader), total=limit_batches, desc=f"Warmup {epoch + 1}/{epochs}", leave=False)

        for i, data in pbar:
            if i >= limit_batches: break

            with torch.no_grad():
                render_pkg_ref = render(data['camera'], fixed_gaussians, data['time'], background, stage="fine")
                ref_image = render_pkg_ref["render"].unsqueeze(0)

            wm_image = inn_model.embed(ref_image, watermark_key)
            restored_image, w_logits = inn_model.extract(wm_image, gt_watermark=watermark_key)

            loss_visual = l1_loss(wm_image, ref_image)
            loss_restore = l1_loss(restored_image, ref_image)
            loss_bits = compute_ber_loss(w_logits, watermark_key)

            loss = loss_visual + loss_bits + loss_restore
            optimizer.zero_grad();
            loss.backward();
            optimizer.step()

            acc = compute_accuracy(w_logits, watermark_key)
            epoch_acc += acc
            pbar.set_postfix({"Acc": f"{acc:.2f}"})

    print("[Phase 1] Warmup complete.\n")
    return inn_model


def train_watermark(opt, dataloader):
    # 1. Load Fixed Reference Model
    print(f"[Init] Loading Fixed Reference Model from {opt.pretrained_model_path}")
    fixed_gaussians = GaussianModel(opt)
    fixed_gaussians.load_ply(os.path.join(opt.pretrained_model_path, "point_cloud.ply"))
    fixed_gaussians.load_model(opt.pretrained_model_path)

    # Freeze Fixed Model
    print("Freezing fixed reference model parameters...")
    params_to_freeze = [
        fixed_gaussians._xyz, fixed_gaussians._features_dc, fixed_gaussians._features_rest,
        fixed_gaussians._scaling, fixed_gaussians._rotation, fixed_gaussians._opacity
    ]
    for param in params_to_freeze:
        if param is not None: param.requires_grad = False
    if hasattr(fixed_gaussians, '_deformation'):
        for param in fixed_gaussians._deformation.parameters(): param.requires_grad = False

    # 2. Load Trainable Model
    print(f"[Init] Loading Trainable Model from {opt.pretrained_model_path}")
    trainable_gaussians = GaussianModel(opt)
    trainable_gaussians.load_ply(os.path.join(opt.pretrained_model_path, "point_cloud.ply"))
    trainable_gaussians.load_model(opt.pretrained_model_path)

    # 3. INN Setup
    inn_model = WatermarkINN(wm_len=opt.wm_len, alpha=0.1).cuda()
    watermark_key = torch.randint(0, 2, (1, opt.wm_len)).float().cuda()
    print(f"Generated Key: {watermark_key[0, :10].cpu().numpy()}...")

    # 4. Warmup
    inn_model = warmup_inn(inn_model, dataloader, watermark_key, fixed_gaussians, epochs=20)

    # 5. Optimizer Setup
    inn_optimizer = optim.Adam(inn_model.parameters(), lr=1e-4)
    inn_model.train()
    trainable_gaussians.training_setup()

    hot_params = ['f_dc', 'f_rest']
    for param_group in trainable_gaussians.optimizer.param_groups:
        if param_group['name'] in hot_params:
            param_group['lr'] *= 0.5
            for param in param_group['params']: param.requires_grad = True
        else:
            param_group['lr'] = 0.0
            for param in param_group['params']: param.requires_grad = False

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    progress_bar = tqdm(range(opt.watermark_iters), desc="[Phase 2] Joint Training")
    ema_psnr, ema_acc, ema_rec = 0.0, 0.0, 0.0
    warned_missing_mask = False

    lambda_ber = 0.1
    lambda_restore = 2.0

    for iteration in progress_bar:
        try:
            data = next(iter(dataloader))
        except StopIteration:
            dataloader_iter = iter(dataloader)
            data = next(dataloader_iter)

        gt_sensor_image = data['camera'].original_image.cuda().unsqueeze(0)
        B, C, H, W = gt_sensor_image.shape

        # --- Mask Logic (Correction) ---
        tool_mask = data['mask'].cuda().unsqueeze(0).unsqueeze(0).float()  # 1 on tool, 0 on bg
        if tool_mask.shape[2] != H or tool_mask.shape[3] != W:
            tool_mask = F.interpolate(tool_mask, size=(H, W), mode='nearest')

        if 'lesion_mask' in data:
            lesion_mask = data['lesion_mask'].cuda().unsqueeze(0).unsqueeze(0).float()  # 1 on lesion
            if lesion_mask.shape[2] != H or lesion_mask.shape[3] != W:
                lesion_mask = F.interpolate(lesion_mask, size=(H, W), mode='nearest')
        else:
            if not warned_missing_mask and iteration == 0:
                print(f"[WARN] No lesion_mask found. Using Ones.")
                warned_missing_mask = True
            lesion_mask = torch.zeros_like(tool_mask)  # Assume no lesion if missing

        # [Corrected] Valid Background = NOT Tool AND NOT Lesion
        # valid_bg_mask = (1 - Tool) * (1 - Lesion)
        valid_bg_mask = (1.0 - tool_mask) * (1.0 - lesion_mask)

        # RONI (Forbidden Area) for embedding calculation
        # To match logic: We embed in valid_bg_mask (after erosion)
        inverted_roni = valid_bg_mask  # This is where we WANT to embed

        erosion_kernel = 21
        # Erode the valid area (equivalent to dilating the forbidden area)
        # Using max_pool on (1 - valid) or min_pool on valid.
        # Original code used max_pool on inverted_roni? No, original was inconsistent.
        # Let's simple erode the valid_bg_mask to stay away from boundaries
        dilated_valid = -F.max_pool2d(-valid_bg_mask, kernel_size=erosion_kernel, stride=1, padding=erosion_kernel // 2)
        # This is strictly "valid area eroded".
        final_embed_mask = dilated_valid

        # Clamp to 0-1 just in case
        final_embed_mask = torch.clamp(final_embed_mask, 0.0, 1.0)

        # --- Forward ---
        trainable_gaussians.optimizer.zero_grad(set_to_none=True)
        inn_optimizer.zero_grad()

        # 1. Generate Reference
        with torch.no_grad():
            render_pkg_ref = render(data['camera'], fixed_gaussians, data['time'], background, stage="fine")
            ref_image = render_pkg_ref["render"].unsqueeze(0)

            # 2. Generate Trainable Render
        render_pkg = render(data['camera'], trainable_gaussians, data['time'], background, stage="fine")
        rendered_image = render_pkg["render"].unsqueeze(0)

        # 3. Prepare Target
        wm_full = inn_model.embed(ref_image, watermark_key)
        # Target: Embed Watermark ONLY in final_embed_mask, else keep Ref
        target_image = wm_full * final_embed_mask + ref_image * (1.0 - final_embed_mask)

        # 4. Extract & Restore
        extractor_input = rendered_image * final_embed_mask
        restored_image, w_logits = inn_model.extract(extractor_input, gt_watermark=watermark_key)

        # --- Loss Calculation [Corrected Masks] ---

        # GS Loss: Rendered should match Target in the Valid Background Area
        # We can strictly constrain this loss to valid_bg_mask to avoid Tool/Lesion interference
        loss_gs = l1_loss(rendered_image * valid_bg_mask, target_image * valid_bg_mask)

        loss_ber = compute_ber_loss(w_logits, watermark_key)

        # Restore Loss: Restored BG should match Reference BG
        loss_rec = l1_loss(restored_image * valid_bg_mask, ref_image * valid_bg_mask)

        loss = loss_gs + lambda_ber * loss_ber + lambda_restore * loss_rec
        loss.backward()

        trainable_gaussians.optimizer.step()
        inn_optimizer.step()

        # --- Metrics [Corrected Masks] ---
        with torch.no_grad():
            acc = compute_accuracy(w_logits, watermark_key)
            # PSNR: Rendered vs Ref (Only in Valid BG)
            cur_psnr = psnr(rendered_image * valid_bg_mask, ref_image * valid_bg_mask).mean().double().item()
            # Rec: Restored vs Ref (Only in Valid BG)
            cur_rec = psnr(restored_image * valid_bg_mask, ref_image * valid_bg_mask).mean().double().item()

            ema_psnr = 0.4 * cur_psnr + 0.6 * ema_psnr
            ema_acc = 0.4 * acc + 0.6 * ema_acc
            ema_rec = 0.4 * cur_rec + 0.6 * ema_rec

        if iteration % 10 == 0:
            progress_bar.set_postfix({
                "PSNR_BG": f"{ema_psnr:.2f}",
                "Rec_BG": f"{ema_rec:.2f}",
                "Acc": f"{ema_acc:.2f}"
            })

    print(f"\nSaving to {opt.workspace}...")
    os.makedirs(opt.workspace, exist_ok=True)
    trainable_gaussians.save(opt.workspace, opt.watermark_iters, "fine")
    ckpt_dir = os.path.join(opt.workspace, "point_cloud", f"iteration_{opt.watermark_iters}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(inn_model.state_dict(), os.path.join(ckpt_dir, "watermark_inn.pth"))
    torch.save(watermark_key, os.path.join(ckpt_dir, "watermark_key.pth"))
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Dataset path")
    parser.add_argument('--pretrained_model_path', type=str, required=True)
    parser.add_argument('--workspace', type=str, default='output/watermarked_gs')
    parser.add_argument('--watermark_iters', type=int, default=5000)
    parser.add_argument('--wm_len', type=int, default=64)
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1])

    # Standard GS Params
    parser.add_argument('--sh_degree', type=int, default=3)
    parser.add_argument('--percent_dense', type=float, default=0.01)
    parser.add_argument('--position_lr_init', type=float, default=0.00016)
    parser.add_argument('--feature_lr', type=float, default=0.0025)
    parser.add_argument('--opacity_lr', type=float, default=0.05)
    parser.add_argument('--scaling_lr', type=float, default=0.005)
    parser.add_argument('--rotation_lr', type=float, default=0.001)
    parser.add_argument('--position_lr_final', type=float, default=0.0000016)
    parser.add_argument('--position_lr_delay_mult', type=float, default=0.01)
    parser.add_argument('--position_lr_max_steps', type=int, default=1000000)
    parser.add_argument('--grid_lr_init', type=float, default=0.00015)
    parser.add_argument('--grid_lr_final', type=float, default=0.000015)
    parser.add_argument('--deformation_lr_init', type=float, default=0.000015)
    parser.add_argument('--deformation_lr_final', type=float, default=0.0000015)
    parser.add_argument('--deformation_lr_delay_mult', type=float, default=0.01)
    parser.add_argument('--deformation_lr_max_steps', type=int, default=1000000)

    opt, _ = parser.parse_known_args()
    seed_everything(0)
    device = torch.device('cuda')
    dataset = EndoDataset(opt, device=device, type='train')
    train_watermark(opt, dataset.dataloader())