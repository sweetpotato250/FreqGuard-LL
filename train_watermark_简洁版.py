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
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr


# ==============================================================================
# SECTION 1: 核心网络定义 (Watermark Core)
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
        x_HH = x1 - x2 + x3 + x4
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
        return self.idwt(x)

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


# ==============================================================================
# SECTION 2: 内部热身函数 (Internal Warmup)
# ==============================================================================

def warmup_inn(inn_model, dataloader, watermark_key, epochs=30):
    print(f"\n[Phase 1] Warming up INN for {epochs} epochs...")
    optimizer = optim.Adam(inn_model.parameters(), lr=1e-3)
    inn_model.train()

    for epoch in range(epochs):
        epoch_acc = 0
        count = 0
        limit_batches = 50

        pbar = tqdm(enumerate(dataloader), total=limit_batches, desc=f"Warmup Epoch {epoch + 1}/{epochs}", leave=False)

        for i, data in pbar:
            if i >= limit_batches: break

            gt_image = data['camera'].original_image.cuda().unsqueeze(0)

            # Forward
            wm_image = inn_model.embed(gt_image, watermark_key)
            _, w_logits = inn_model.extract(wm_image)

            # Loss
            loss = l1_loss(wm_image, gt_image) + compute_ber_loss(w_logits, watermark_key)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = compute_accuracy(w_logits, watermark_key)
            epoch_acc += acc
            count += 1
            pbar.set_postfix({"Acc": f"{acc:.2f}"})

        print(f"  > Epoch {epoch + 1} Avg Acc: {epoch_acc / count:.4f}")

    print("[Phase 1] Warmup complete. INN is ready.\n")
    return inn_model


# ==============================================================================
# SECTION 3: 主训练逻辑 (Main Loop)
# ==============================================================================

def train_watermark(opt, dataloader, gaussians):
    # 1. 加载 EndoGS 预训练模型
    print(f"Loading Gaussians from {opt.pretrained_model_path}")
    gaussians.load_ply(os.path.join(opt.pretrained_model_path, "point_cloud.ply"))
    gaussians.load_model(opt.pretrained_model_path)

    # 2. 初始化 INN
    print(f"Initializing INN (wm_len={opt.wm_len})...")
    inn_model = WatermarkINN(wm_len=opt.wm_len).cuda()

    # 3. 生成密钥
    watermark_key = torch.randint(0, 2, (1, opt.wm_len)).float().cuda()
    print(f"Generated Key: {watermark_key[0, :10].cpu().numpy()}...")

    # 4. 热身
    inn_model = warmup_inn(inn_model, dataloader, watermark_key, epochs=30)

    # 5. 冻结 INN
    inn_model.eval()
    for param in inn_model.parameters():
        param.requires_grad = False

    # 6. 设置 Gaussian 优化器并冻结几何参数
    print("Setting up Gaussian optimizer and freezing geometry (xyz, rotation, scaling)...")
    gaussians.training_setup()

    # 显式冻结几何参数的梯度计算
    gaussians._xyz.requires_grad = False
    gaussians._rotation.requires_grad = False
    gaussians._scaling.requires_grad = False

    # 确保颜色和不透明度参数是可训练的
    gaussians._features_dc.requires_grad = True
    gaussians._features_rest.requires_grad = True
    gaussians._opacity.requires_grad = True

    # 调整优化器中的学习率（将不需要更新的参数 LR 设为 0）
    frozen_names = ['xyz', 'rotation', 'scaling']
    for group in gaussians.optimizer.param_groups:
        if group['name'] in frozen_names:
            group['lr'] = 0.0
        else:
            # 仅对微调参数 (SH, opacity) 降低学习率，保持稳定性
            group['lr'] *= 0.5

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = 0
    max_iter = opt.watermark_iters
    progress_bar = tqdm(range(iter_start, max_iter), desc="[Phase 2] GS Fine-tuning")

    ema_psnr, ema_acc, ema_loss = 0.0, 0.0, 0.0
    lambda_ber = 0.1

    # 标记是否警告过
    warned_missing_mask = False

    for iteration in progress_bar:
        try:
            data = next(iter(dataloader))
        except StopIteration:
            dataloader_iter = iter(dataloader)
            data = next(dataloader_iter)

        gt_image = data['camera'].original_image.cuda().unsqueeze(0)

        # tool_mask: 1=Tissue, 0=Tool
        tool_mask = data['mask'].cuda().unsqueeze(0).unsqueeze(0)

        # 安全获取病变掩码
        if 'lesion_mask' in data:
            lesion_mask = data['lesion_mask'].cuda().unsqueeze(0).unsqueeze(0)
        else:
            if not warned_missing_mask and iteration == 0:
                print("\n[WARNING] 'lesion_mask' not found in dataloader! Fallback to ONES (Assuming no lesions).")
                warned_missing_mask = True
            # Default to all ones (All safe/Non-Lesion)
            lesion_mask = torch.ones_like(tool_mask)

        # --- 定义 RONI (非病变区域) ---
        # 既是非器械 (1)，也是非病变 (1) => 结果为 1
        roni_mask = tool_mask * lesion_mask

        # --- 腐蚀操作 (去除边缘伪影) ---
        erosion_kernel = 21
        inverted_roni = 1.0 - roni_mask
        dilated_inverted = F.max_pool2d(inverted_roni, kernel_size=erosion_kernel, stride=1,
                                        padding=erosion_kernel // 2)
        final_embed_mask = 1.0 - dilated_inverted

        # --- 渲染 ---
        gaussians.optimizer.zero_grad(set_to_none=True)
        render_pkg = render(data['camera'], gaussians, data['time'], background, stage="fine")
        rendered_image = render_pkg["render"].unsqueeze(0)

        # --- 构造目标图像 ---
        with torch.no_grad():
            wm_full = inn_model.embed(gt_image, watermark_key)

        # 物理拼接: 只在最终掩码区用水印图
        target_image = wm_full * final_embed_mask + gt_image * (1.0 - final_embed_mask)

        # --- Loss ---
        loss_recon = l1_loss(rendered_image * tool_mask, target_image * tool_mask)
        _, w_logits = inn_model.extract(rendered_image)
        loss_ber = compute_ber_loss(w_logits, watermark_key)

        loss = loss_recon + lambda_ber * loss_ber
        loss.backward()
        gaussians.optimizer.step()

        # --- Logging ---
        with torch.no_grad():
            acc = compute_accuracy(w_logits, watermark_key)
            cur_psnr = psnr(rendered_image * tool_mask, gt_image * tool_mask).mean().double().item()

            ema_psnr = 0.4 * cur_psnr + 0.6 * ema_psnr
            ema_acc = 0.4 * acc + 0.6 * ema_acc
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss

        if iteration % 10 == 0:
            progress_bar.set_postfix({
                "Loss": f"{ema_loss:.4f}",
                "PSNR": f"{ema_psnr:.2f}",
                "Acc": f"{ema_acc:.2f}"
            })

    # 保存结果
    if opt.workspace is not None:
        os.makedirs(opt.workspace, exist_ok=True)

    print(f"Saving models to {opt.workspace}...")
    gaussians.save(opt.workspace, max_iter, "fine")

    torch.save(inn_model.state_dict(), os.path.join(opt.workspace, "watermark_inn.pth"))
    torch.save(watermark_key, os.path.join(opt.workspace, "watermark_key.pth"))
    print("Training Complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Dataset path")
    parser.add_argument('--pretrained_model_path', type=str, required=True, help="Path to clean pretrained model")
    parser.add_argument('--workspace', type=str, default='output/watermarked_gs', help="Where to save result")
    parser.add_argument('--watermark_iters', type=int, default=5000, help="Number of fine-tuning iterations")
    parser.add_argument('--wm_len', type=int, default=64, help="Watermark length")
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1])

    # GS 默认参数
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