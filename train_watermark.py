import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import argparse
from tqdm import tqdm
import numpy as np
import random  # [新增] 用于随机攻击选择

# Import EndoGS core modules
from gaussian_core.provider import EndoDataset
from gaussian_core.utils import seed_everything
from gaussian_core.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss
from utils.image_utils import psnr

# [新增] 引入攻击模块 (确保 robustness_utils.py 在同级目录下)
from robustness_utils import RobustnessAttacker


# ==============================================================================
# SECTION 1: INN Core Classes (保持 Code A 的结构不变)
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
    def __init__(self, in_channels=12, steps=4, wm_len=64, alpha=0.1, subbands="all"):
        super(WatermarkINN, self).__init__()
        self.dwt = DWT()
        self.idwt = IDWT()
        self.wm_len = wm_len
        self.alpha = alpha

        # --- Subband Selection Logic ---
        self.c_per_subband = in_channels // 4
        subband_order = ['LL', 'HL', 'LH', 'HH']

        if subbands.lower() == 'all':
            selected_names = subband_order
        else:
            selected_names = [s.strip() for s in subbands.split(',')]

        self.active_indices = []
        self.passive_indices = []

        for i, name in enumerate(subband_order):
            indices = list(range(i * self.c_per_subband, (i + 1) * self.c_per_subband))
            if name in selected_names:
                self.active_indices.extend(indices)
            else:
                self.passive_indices.extend(indices)

        self.register_buffer('active_idx', torch.tensor(self.active_indices, dtype=torch.long))
        self.register_buffer('passive_idx', torch.tensor(self.passive_indices, dtype=torch.long))

        self.inn_channels = len(self.active_indices)
        print(f"[WatermarkINN] Subbands: {subbands} | Active Channels: {self.inn_channels}")

        self.layers = nn.ModuleList([CouplingLayer(self.inn_channels) for _ in range(steps)])
        self.wm_projector = nn.Sequential(
            nn.Linear(wm_len, self.inn_channels),
            nn.ReLU(),
            nn.Linear(self.inn_channels, self.inn_channels)
        )
        self.wm_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.inn_channels, 128),
            nn.ReLU(),
            nn.Linear(128, wm_len)
        )

    def get_wm_feature(self, watermark, dims):
        return self.wm_projector(watermark).unsqueeze(-1).unsqueeze(-1).expand(*dims)

    def inn_forward(self, x):
        for layer in self.layers:
            x = layer(x, reverse=False)
        return x

    def inn_inverse(self, z):
        for layer in reversed(self.layers):
            z = layer(z, reverse=True)
        return z

    def split_features(self, coeffs):
        x_active = torch.index_select(coeffs, 1, self.active_idx)
        if len(self.passive_indices) > 0:
            x_passive = torch.index_select(coeffs, 1, self.passive_idx)
        else:
            x_passive = None
        return x_active, x_passive

    def merge_features(self, x_active, x_passive, original_shape):
        if x_passive is None:
            return x_active
        out = torch.zeros(original_shape, device=x_active.device, dtype=x_active.dtype)
        out.index_copy_(1, self.active_idx, x_active)
        out.index_copy_(1, self.passive_idx, x_passive)
        return out

    def embed(self, image, watermark):
        coeffs = self.dwt(image)
        z_active, z_passive = self.split_features(coeffs)

        z_transformed = self.inn_forward(z_active)
        wm_feat = self.get_wm_feature(watermark, z_transformed.shape)
        z_watermarked = z_transformed + self.alpha * wm_feat

        x_active_restored = self.inn_inverse(z_watermarked)
        coeffs_watermarked = self.merge_features(x_active_restored, z_passive, coeffs.shape)
        return self.idwt(coeffs_watermarked)

    def extract(self, watermarked_image, gt_watermark=None):
        coeffs_wm = self.dwt(watermarked_image)
        z_active, z_passive = self.split_features(coeffs_wm)

        z_rec = self.inn_forward(z_active)
        w_logits = self.wm_extractor(z_rec)

        w_to_subtract = gt_watermark if gt_watermark is not None else (torch.sigmoid(w_logits) > 0.5).float()
        z_clean_active = z_rec - self.alpha * self.get_wm_feature(w_to_subtract, z_rec.shape)

        x_active_clean = self.inn_inverse(z_clean_active)
        coeffs_clean = self.merge_features(x_active_clean, z_passive, coeffs_wm.shape)

        return self.idwt(coeffs_clean), w_logits


def compute_ber_loss(w_pred_logits, w_gt):
    return F.binary_cross_entropy_with_logits(w_pred_logits, w_gt)


def compute_accuracy(w_pred_logits, w_gt):
    return ((torch.sigmoid(w_pred_logits) > 0.5).float() == w_gt).float().mean().item()


# ==============================================================================
# SECTION 2: Training Logic (Modified for Robustness)
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
    # -----------------------------------------------------------
    # 1. Load Fixed Reference Model
    # -----------------------------------------------------------
    print(f"[Init] Loading Fixed Reference Model from {opt.pretrained_model_path}")
    fixed_gaussians = GaussianModel(opt)
    fixed_gaussians.load_ply(os.path.join(opt.pretrained_model_path, "point_cloud.ply"))
    fixed_gaussians.load_model(opt.pretrained_model_path)

    # Freeze params
    print("Freezing fixed reference model parameters...")
    params_to_freeze = [fixed_gaussians._xyz, fixed_gaussians._features_dc, fixed_gaussians._features_rest,
                        fixed_gaussians._scaling, fixed_gaussians._rotation, fixed_gaussians._opacity]
    for param in params_to_freeze:
        if param is not None: param.requires_grad = False
    if hasattr(fixed_gaussians, '_deformation'):
        for param in fixed_gaussians._deformation.parameters(): param.requires_grad = False

    # -----------------------------------------------------------
    # 2. Load Trainable Model
    # -----------------------------------------------------------
    print(f"[Init] Loading Trainable Model from {opt.pretrained_model_path}")
    trainable_gaussians = GaussianModel(opt)
    trainable_gaussians.load_ply(os.path.join(opt.pretrained_model_path, "point_cloud.ply"))
    trainable_gaussians.load_model(opt.pretrained_model_path)

    # 3. INN Setup
    inn_model = WatermarkINN(wm_len=opt.wm_len, alpha=0.1, subbands=opt.subbands).cuda()
    watermark_key = torch.randint(0, 2, (1, opt.wm_len)).float().cuda()
    print(f"Generated Key: {watermark_key[0, :10].cpu().numpy()}...")

    # 4. Warmup
    inn_model = warmup_inn(inn_model, dataloader, watermark_key, fixed_gaussians, epochs=20)

    # 5. Optimizer Setup
    inn_optimizer = optim.Adam(inn_model.parameters(), lr=1e-4)
    inn_model.train()
    trainable_gaussians.training_setup()

    # Only fine-tune color features
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

    # [新增] 攻击参数
    aug_prob = 0.5  # 50% 概率受到攻击

    for iteration in progress_bar:
        try:
            data = next(iter(dataloader))
        except StopIteration:
            dataloader_iter = iter(dataloader)
            data = next(dataloader_iter)

        gt_sensor_image = data['camera'].original_image.cuda().unsqueeze(0)
        B, C, H, W = gt_sensor_image.shape

        # --- Mask Logic ---
        tool_mask = data['mask'].cuda().unsqueeze(0).unsqueeze(0).float()
        if tool_mask.shape[2] != H or tool_mask.shape[3] != W:
            tool_mask = F.interpolate(tool_mask, size=(H, W), mode='nearest')

        if 'lesion_mask' in data:
            lesion_mask = data['lesion_mask'].cuda().unsqueeze(0).unsqueeze(0).float()
            if lesion_mask.shape[2] != H or lesion_mask.shape[3] != W:
                lesion_mask = F.interpolate(lesion_mask, size=(H, W), mode='nearest')
        else:
            if not warned_missing_mask and iteration == 0:
                print(f"[WARN] No lesion_mask found. Using Ones.")
                warned_missing_mask = True
            lesion_mask = torch.ones_like(tool_mask)

        roni_mask = tool_mask * lesion_mask
        erosion_kernel = 21
        inverted_roni = 1.0 - roni_mask
        dilated_inverted = F.max_pool2d(inverted_roni, kernel_size=erosion_kernel, stride=1,
                                        padding=erosion_kernel // 2)
        final_embed_mask = 1.0 - dilated_inverted

        # --- Forward ---
        trainable_gaussians.optimizer.zero_grad(set_to_none=True)
        inn_optimizer.zero_grad()

        # 1. Reference
        with torch.no_grad():
            render_pkg_ref = render(data['camera'], fixed_gaussians, data['time'], background, stage="fine")
            ref_image = render_pkg_ref["render"].unsqueeze(0)

        # 2. Trainable Render (Watermarked by GS optimization)
        render_pkg = render(data['camera'], trainable_gaussians, data['time'], background, stage="fine")
        rendered_image = render_pkg["render"].unsqueeze(0)

        # 3. Target (INN Embed for Visual Supervision)
        wm_full = inn_model.embed(ref_image, watermark_key)
        target_image = wm_full * final_embed_mask + ref_image * (1.0 - final_embed_mask)

        # 4. Robustness Extraction (加入攻击层!)
        # 复制一份用于提取，保持原始 rendered_image 用于计算视觉 Loss
        extractor_input_raw = rendered_image.clone()

        # [修改] 训练期间施加攻击
        is_attacked = False
        if torch.rand(1).item() < aug_prob:
            is_attacked = True
            aug_type = torch.randint(0, 5, (1,)).item()

            # 使用较温和的参数以防止色偏
            if aug_type == 0:  # Noise
                extractor_input_raw = RobustnessAttacker.attack_noise(extractor_input_raw, std=0.05)
            elif aug_type == 1:  # Scaling (温和缩放，0.8 而非 0.25)
                # 注意：DWT 对分辨率敏感，缩放后需要 resize 回来，或者让网络适应模糊
                extractor_input_raw = RobustnessAttacker.attack_scaling(extractor_input_raw, scale=0.8)
                # 插值回原尺寸以便和 mask 对齐 (这对 DWT 也是一种攻击，相当于 blur)
                extractor_input_raw = F.interpolate(extractor_input_raw, size=(H, W), mode='bilinear')
            elif aug_type == 2:  # Blur
                extractor_input_raw = RobustnessAttacker.attack_blur(extractor_input_raw, sigma=0.5)
            elif aug_type == 3:  # Brightness
                extractor_input_raw = RobustnessAttacker.attack_brightness(extractor_input_raw, factor=1.2)
            elif aug_type == 4:  # Crop (温和裁剪)
                # 简单 Crop 可能会导致 mask 不对齐，但在训练中作为一种 noise 引入
                extractor_input_raw = RobustnessAttacker.attack_crop(extractor_input_raw, crop_percent=0.1)
                extractor_input_raw = F.interpolate(extractor_input_raw, size=(H, W), mode='bilinear')

        # 应用 mask 准备提取
        extractor_input = extractor_input_raw * final_embed_mask

        # 提取水印
        restored_image, w_logits = inn_model.extract(extractor_input, gt_watermark=watermark_key)

        # --- Loss ---
        loss_gs = l1_loss(rendered_image * tool_mask, target_image * tool_mask)
        loss_ber = compute_ber_loss(w_logits, watermark_key)

        # 如果受到了攻击，提取出的图像肯定和原图不像，所以仅在未攻击时强烈约束 loss_rec
        if is_attacked:
            loss_rec = l1_loss(restored_image * final_embed_mask, ref_image * final_embed_mask) * 0.1
        else:
            loss_rec = l1_loss(restored_image * final_embed_mask, ref_image * final_embed_mask)

        loss = loss_gs + lambda_ber * loss_ber + lambda_restore * loss_rec
        loss.backward()

        trainable_gaussians.optimizer.step()
        inn_optimizer.step()

        # --- Metrics ---
        with torch.no_grad():
            acc = compute_accuracy(w_logits, watermark_key)
            cur_psnr = psnr(rendered_image * tool_mask, ref_image * tool_mask).mean().double().item()
            cur_rec = psnr(restored_image * final_embed_mask, ref_image * final_embed_mask).mean().double().item()

            ema_psnr = 0.4 * cur_psnr + 0.6 * ema_psnr
            ema_acc = 0.4 * acc + 0.6 * ema_acc
            ema_rec = 0.4 * cur_rec + 0.6 * ema_rec

        if iteration % 10 == 0:
            progress_bar.set_postfix({
                "PSNR": f"{ema_psnr:.2f}",
                "Rec": f"{ema_rec:.2f}",
                "Acc": f"{ema_acc:.2f}"
            })

    print(f"\nSaving to {opt.workspace}...")
    os.makedirs(opt.workspace, exist_ok=True)
    trainable_gaussians.save(opt.workspace, opt.watermark_iters, "fine")
    ckpt_dir = os.path.join(opt.workspace, "point_cloud", f"iteration_{opt.watermark_iters}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(inn_model.state_dict(), os.path.join(ckpt_dir, "watermark_inn.pth"))
    torch.save(watermark_key, os.path.join(ckpt_dir, "watermark_key.pth"))

    # [新增] 保存训练配置，方便 Test 读取
    with open(os.path.join(ckpt_dir, "train_config.txt"), "w") as f:
        f.write(f"subbands: {opt.subbands}\n")

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Dataset path")
    parser.add_argument('--pretrained_model_path', type=str, required=True,
                        help="Path to clean pretrained model (Reference)")
    parser.add_argument('--workspace', type=str, default='output/watermarked_gs')
    parser.add_argument('--watermark_iters', type=int, default=5000)
    parser.add_argument('--wm_len', type=int, default=64)
    # [Code A 特性] 支持子带选择
    parser.add_argument('--subbands', type=str, default='all',
                        help='Subbands to embed: "all", "LL", "HL,LH", etc.')
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1])

    # GS Params
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