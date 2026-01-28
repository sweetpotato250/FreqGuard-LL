import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from tqdm import tqdm
import numpy as np

# 导入必要的模块
from gaussian_core.provider import EndoDataset
from gaussian_core.utils import *
from gaussian_core.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim  # [Added] 导入 ssim
from utils.image_utils import psnr  # [Added] 导入 psnr


# ==========================================
# Part 1: DWT & IDWT (Haar Wavelet)
# ==========================================
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        # x: [B, C, H, W]
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
        # x: [B, 4*C, H/2, W/2]
        in_batch, in_channel, in_height, in_width = x.size()
        out_channel = int(in_channel / 4)
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


# ==========================================
# Part 2: Invertible Neural Network (INN) Block
# ==========================================
class CouplingLayer(nn.Module):
    def __init__(self, in_channels, mid_channels=64):
        super(CouplingLayer, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels // 2, mid_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_channels // 2, 3, 1, 1)
        )

    def forward(self, x, reverse=False):
        x1, x2 = torch.chunk(x, 2, dim=1)
        if not reverse:
            y1 = x1
            y2 = x2 + self.net(x1)
            return torch.cat([y1, y2], dim=1)
        else:
            y1 = x1
            y2 = x2 - self.net(x1)
            return torch.cat([y1, y2], dim=1)


class WatermarkINN(nn.Module):
    def __init__(self, in_channels=12, steps=3, wm_len=64):
        super(WatermarkINN, self).__init__()
        self.dwt = DWT()
        self.idwt = IDWT()
        self.wm_len = wm_len
        self.layers = nn.ModuleList([CouplingLayer(in_channels) for _ in range(steps)])
        self.wm_projection = nn.Linear(wm_len, in_channels)
        self.wm_extract_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, wm_len)
        )

    def embed(self, image, watermark):
        coeffs = self.dwt(image)
        wm_feat = self.wm_projection(watermark).unsqueeze(-1).unsqueeze(-1)
        x = coeffs + wm_feat
        for layer in self.layers:
            x = layer(x, reverse=False)
        watermarked_image = self.idwt(x)
        return watermarked_image

    def extract(self, watermarked_image):
        coeffs = self.dwt(watermarked_image)
        x = coeffs
        for layer in reversed(self.layers):
            x = layer(x, reverse=True)
        w_pred_logits = self.wm_extract_head(x)
        restored_image = self.idwt(x)
        return restored_image, w_pred_logits


def compute_ber_loss(w_pred_logits, w_gt):
    return F.binary_cross_entropy_with_logits(w_pred_logits, w_gt)


def compute_accuracy(w_pred_logits, w_gt):
    w_pred_bits = (torch.sigmoid(w_pred_logits) > 0.5).float()
    correct_bits = (w_pred_bits == w_gt).float()
    return correct_bits.mean().item()


# ==========================================
# Part 3: Training Loop
# ==========================================
def train_watermark(opt, dataloader, gaussians):
    # 1. Load Pre-trained Model
    print(f"Loading pre-trained model from {opt.pretrained_model_path}")
    if not os.path.exists(os.path.join(opt.pretrained_model_path, "point_cloud.ply")):
        raise FileNotFoundError(f"point_cloud.ply not found in {opt.pretrained_model_path}")

    gaussians.load_ply(os.path.join(opt.pretrained_model_path, "point_cloud.ply"))
    gaussians.load_model(opt.pretrained_model_path)

    # 2. Setup Watermark Model
    inn_model = WatermarkINN().cuda()
    optimizer_inn = torch.optim.Adam(inn_model.parameters(), lr=1e-4)

    # 3. Setup Gaussian Optimizer
    gaussians.training_setup()
    for param_group in gaussians.optimizer.param_groups:
        param_group['lr'] *= 0.1

        # 4. Define Watermark Key
    watermark_key = torch.randint(0, 2, (1, 64)).float().cuda()
    print(f"Generated Watermark Key: {watermark_key[0, :10]}... (first 10 bits)")

    iter_start = 0
    max_iter = opt.watermark_iters
    progress_bar = tqdm(range(iter_start, max_iter), desc="Watermark Fine-tuning")

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Hyperparameters
    lambda_ber = 0.5
    lambda_w_rec = 1.0

    # 用于平滑日志显示的 EMA (Exponential Moving Average)
    ema_psnr = 0.0
    ema_ssim = 0.0
    ema_acc = 0.0
    ema_loss = 0.0

    for iteration in range(iter_start, max_iter):
        try:
            data = next(iter(dataloader))
        except StopIteration:
            dataloader_iter = iter(dataloader)
            data = next(dataloader_iter)

        gt_image = data['camera'].original_image.cuda().unsqueeze(0)
        mask = data['mask'].cuda().unsqueeze(0).unsqueeze(0)
        valid_mask = mask

        # --- Step A: INN Update ---
        optimizer_inn.zero_grad()
        wm_image = inn_model.embed(gt_image, watermark_key)
        restored_image, w_pred_logits = inn_model.extract(wm_image)

        loss_imperceptibility = l1_loss(wm_image * valid_mask, gt_image * valid_mask)
        loss_w_extract = compute_ber_loss(w_pred_logits, watermark_key)

        loss_inn = loss_imperceptibility + lambda_w_rec * loss_w_extract
        loss_inn.backward()
        optimizer_inn.step()

        # --- Step B: Gaussian Update ---
        gaussians.optimizer.zero_grad(set_to_none=True)
        render_pkg = render(data['camera'], gaussians, data['time'], background, stage="fine")
        rendered_image = render_pkg["render"].unsqueeze(0)

        w_pred_logits_render = inn_model.extract(rendered_image)[1]

        target_image = wm_image.detach()
        loss_recon = l1_loss(rendered_image * valid_mask, target_image * valid_mask)
        loss_ber_gs = compute_ber_loss(w_pred_logits_render, watermark_key)

        # 总损失
        loss_gs = loss_recon + lambda_ber * loss_ber_gs

        loss_gs.backward()
        gaussians.optimizer.step()

        # --- Metrics Calculation ---
        with torch.no_grad():
            # 1. Accuracy
            current_acc = compute_accuracy(w_pred_logits_render, watermark_key)

            # 2. PSNR & SSIM (Rendered vs Original GT) - 反映画质影响
            # 限制在有效区域或全图计算均可，这里计算 mask 区域更有意义，但标准库通常计算全图。
            # 简单起见，我们计算 masked 后的图像指标，或者直接计算全图。
            # 为了反映真实感知，计算 masked 区域的 PSNR 会更准确地反映组织变化。
            # 这里简单做：(Render * Mask) vs (GT * Mask)

            masked_render = rendered_image * valid_mask
            masked_gt = gt_image * valid_mask

            current_psnr = psnr(masked_render, masked_gt).mean().double().item()
            current_ssim = ssim(masked_render, masked_gt).mean().item()

            # EMA Updates
            ema_psnr = 0.4 * current_psnr + 0.6 * ema_psnr
            ema_ssim = 0.4 * current_ssim + 0.6 * ema_ssim
            ema_acc = 0.4 * current_acc + 0.6 * ema_acc
            ema_loss = 0.4 * loss_gs.item() + 0.6 * ema_loss

        if iteration % 10 == 0:
            progress_bar.set_postfix({
                "PSNR": f"{ema_psnr:.2f}",
                "SSIM": f"{ema_ssim:.4f}",
                "Acc": f"{ema_acc:.4f}",
                "Loss": f"{ema_loss:.5f}"
            })

    # Save
    if opt.workspace is not None:
        os.makedirs(opt.workspace, exist_ok=True)

    print(f"Saving watermarked model to {opt.workspace}")
    gaussians.save(opt.workspace, max_iter, "fine")
    torch.save(inn_model.state_dict(), os.path.join(opt.workspace, "watermark_inn.pth"))
    torch.save(watermark_key, os.path.join(opt.workspace, "watermark_key.pth"))
    print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Path to dataset")
    parser.add_argument('--pretrained_model_path', type=str, required=True,
                        help="Path to clean pretrained model directory")
    parser.add_argument('--workspace', type=str, default='output/watermarked_model', help="Where to save result")
    parser.add_argument('--watermark_iters', type=int, default=5000, help="Number of fine-tuning iterations")
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1])

    # Gaussian params
    parser.add_argument('--sh_degree', type=int, default=3)
    parser.add_argument('--percent_dense', type=float, default=0.01)
    parser.add_argument('--position_lr_init', type=float, default=0.00016)
    parser.add_argument('--position_lr_final', type=float, default=0.0000016)
    parser.add_argument('--position_lr_delay_mult', type=float, default=0.01)
    parser.add_argument('--position_lr_max_steps', type=int, default=1000000)
    parser.add_argument('--grid_lr_init', type=float, default=0.00015)
    parser.add_argument('--grid_lr_final', type=float, default=0.000015)
    parser.add_argument('--deformation_lr_init', type=float, default=0.000015)
    parser.add_argument('--deformation_lr_final', type=float, default=0.0000015)
    parser.add_argument('--deformation_lr_delay_mult', type=float, default=0.01)
    parser.add_argument('--deformation_lr_max_steps', type=int, default=1000000)
    parser.add_argument('--feature_lr', type=float, default=0.0025)
    parser.add_argument('--opacity_lr', type=float, default=0.05)
    parser.add_argument('--scaling_lr', type=float, default=0.005)
    parser.add_argument('--rotation_lr', type=float, default=0.001)

    opt = parser.parse_args()

    seed_everything(0)

    gaussians = GaussianModel(opt)
    device = torch.device('cuda')
    dataset = EndoDataset(opt, device=device, type='train')
    dataloader = dataset.dataloader()

    train_watermark(opt, dataloader, gaussians)