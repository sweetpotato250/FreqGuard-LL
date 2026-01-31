import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from tqdm import tqdm
import numpy as np

# 假设这些模块都在你的路径下
from gaussian_core.provider import EndoDataset
from gaussian_core.utils import seed_everything
from gaussian_core.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr


# ==============================================================================
# SECTION 1: 参数级 INN (Parameter INN)
# ==============================================================================

class PointCouplingLayer(nn.Module):
    def __init__(self, in_channels, mid_channels=64):
        super(PointCouplingLayer, self).__init__()
        self.split_len1 = in_channels // 2
        self.split_len2 = in_channels - self.split_len1

        # 使用 1D 卷积，相当于对每个高斯点独立做 MLP，但共享权重
        self.net = nn.Sequential(
            nn.Conv1d(self.split_len1, mid_channels, 1),
            nn.ReLU(),
            nn.Conv1d(mid_channels, mid_channels, 1),
            nn.ReLU(),
            nn.Conv1d(mid_channels, self.split_len2 * 2, 1)
        )

    def forward(self, x, reverse=False):
        # x shape: [Batch=1, Channels, N_points]
        x1, x2 = torch.split(x, [self.split_len1, self.split_len2], dim=1)
        st = self.net(x1)
        s, t = torch.chunk(st, 2, dim=1)

        # 缩放因子 s 必须受限，防止数值爆炸
        s = torch.sigmoid(s + 2) + 1e-6

        if not reverse:
            return torch.cat([x1, s * x2 + t], dim=1)
        else:
            return torch.cat([x1, (x2 - t) / s], dim=1)


class ParameterINN(nn.Module):
    def __init__(self, in_channels=45, steps=4, wm_len=64, alpha=0.1):
        """
        in_channels: 45 (SH degrees 1-3: 15 coeffs * 3 RGB)
        """
        super(ParameterINN, self).__init__()
        self.layers = nn.ModuleList([PointCouplingLayer(in_channels) for _ in range(steps)])
        self.wm_len = wm_len
        self.alpha = alpha

        # Watermark Projector: [wm_len] -> [in_channels]
        # 用于将水印广播到每个点的特征空间
        self.wm_projector = nn.Sequential(
            nn.Linear(wm_len, 128),
            nn.ReLU(),
            nn.Linear(128, in_channels)
        )

        # Extractor: [in_channels] -> [wm_len]
        # 从参数的统计特征中提取水印 (Global Average Pooling)
        self.wm_extractor = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, wm_len)
        )

    def get_wm_feature(self, watermark, n_points):
        # watermark: [1, wm_len]
        # output: [1, C, N]
        feat = self.wm_projector(watermark).unsqueeze(-1)  # [1, C, 1]
        return feat.expand(1, -1, n_points)

    def forward_inn(self, x):
        for layer in self.layers:
            x = layer(x, reverse=False)
        return x

    def inverse_inn(self, z):
        for layer in reversed(self.layers):
            z = layer(z, reverse=True)
        return z

    def embed(self, params_clean, watermark):
        """
        params_clean: [N, 45] Tensor
        watermark: [1, wm_len] Tensor
        Return: params_watermarked [N, 45]
        """
        # Reshape to [1, 45, N] for Conv1d
        x = params_clean.unsqueeze(0).permute(0, 2, 1)
        N = x.shape[2]

        # 1. Forward to Latent
        z = self.forward_inn(x)

        # 2. Add Watermark
        wm_feat = self.get_wm_feature(watermark, N)
        z_wm = z + self.alpha * wm_feat

        # 3. Inverse to Parameter Space
        x_wm = self.inverse_inn(z_wm)

        # Reshape back to [N, 45]
        return x_wm.permute(0, 2, 1).squeeze(0)

    def extract_and_restore(self, params_wm, gt_watermark=None):
        """
        params_wm: [N, 45]
        Return: params_restored, w_logits
        """
        x_wm = params_wm.unsqueeze(0).permute(0, 2, 1)
        N = x_wm.shape[2]

        # 1. Forward to Latent (Watermarked Z)
        z_wm = self.forward_inn(x_wm)

        # 2. Extract Watermark (Global Avg Pool over points)
        # z_wm shape: [1, C, N] -> avg -> [1, C]
        z_global = torch.mean(z_wm, dim=2)
        w_logits = self.wm_extractor(z_global)

        # 3. Subtract Watermark
        if gt_watermark is not None:
            w_target = gt_watermark
        else:
            w_target = (torch.sigmoid(w_logits) > 0.5).float()

        wm_feat = self.get_wm_feature(w_target, N)
        z_clean = z_wm - self.alpha * wm_feat

        # 4. Inverse to Clean Parameters
        x_clean = self.inverse_inn(z_clean)

        return x_clean.permute(0, 2, 1).squeeze(0), w_logits


def compute_ber_loss(w_pred_logits, w_gt):
    return nn.functional.binary_cross_entropy_with_logits(w_pred_logits, w_gt)


def compute_accuracy(w_pred_logits, w_gt):
    return ((torch.sigmoid(w_pred_logits) > 0.5).float() == w_gt).float().mean().item()


# ==============================================================================
# SECTION 2: 训练逻辑
# ==============================================================================

def train_param_watermark(opt, dataloader, gaussians):
    # 1. 加载预训练高斯模型
    print(f"[INFO] Loading Gaussians from {opt.pretrained_model_path}")
    gaussians.load_ply(os.path.join(opt.pretrained_model_path, "point_cloud.ply"))

    # 2. 获取并锁定 Clean Features (SH Rest)
    # SH Rest 通常是 [N, 15, 3] -> Flatten to [N, 45]
    # 我们不仅需要值，还需要它是 Tensor 以便传入网络
    with torch.no_grad():
        # 获取 f_rest, 形状通常是 [N, 45] (15*3 flattened by wrapper) 或 [N, 15, 3]
        # 在 GaussianModel 中，_features_rest 是 [N, 15, 3]
        clean_sh_original = gaussians._features_rest.clone().detach()
        N_points = clean_sh_original.shape[0]
        # Flatten: [N, 15, 3] -> [N, 45]
        clean_sh_flat = clean_sh_original.view(N_points, -1)

        print(f"[INFO] Processing {N_points} points. SH Feature shape: {clean_sh_flat.shape}")

    # 3. 初始化 INN
    inn_model = ParameterINN(in_channels=45, wm_len=opt.wm_len, alpha=0.1).cuda()
    inn_optimizer = optim.Adam(inn_model.parameters(), lr=1e-4)

    # 生成 Key
    watermark_key = torch.randint(0, 2, (1, opt.wm_len)).float().cuda()
    print(f"[INFO] Watermark Key: {watermark_key[0, :10].cpu().numpy()}...")

    # 4. 训练设置
    # 注意：我们完全冻结 GaussianModel，只训练 INN
    gaussians.training_setup(opt)  # 修改点：确保传入 opt
    for param_group in gaussians.optimizer.param_groups:
        param_group['lr'] = 0.0  # 强制设为0，确保 GS 不更新

    # 准备背景
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # 进度条
    progress_bar = tqdm(range(opt.watermark_iters), desc="Training Parameter INN")
    ema_psnr = 0.0
    ema_acc = 0.0

    for iteration in progress_bar:
        try:
            data = next(iter(dataloader))
        except StopIteration:
            dataloader_iter = iter(dataloader)
            data = next(dataloader_iter)

        gt_image = data['camera'].original_image.cuda().unsqueeze(0)

        inn_optimizer.zero_grad()

        # --- Step 1: INN Embedding ---
        # 输入: 干净的参数 (Memory)
        # 输出: 带水印的参数
        wm_sh_flat = inn_model.embed(clean_sh_flat, watermark_key)

        # --- Step 2: Render with Override ---
        # 这是一个关键 Hack：我们将 GS 的参数临时替换为 INN 的输出
        # reshape back to [N, 15, 3]
        wm_sh_reshaped = wm_sh_flat.view(N_points, 15, 3)

        # 保存原始指针 (虽然是 detached 的，但为了逻辑严谨)
        original_handle = gaussians._features_rest

        # *替换*: 将 Parameter 属性指向我们的计算图节点
        # 这样 render() 里的操作就会把梯度传回 wm_sh_reshaped -> inn_model
        gaussians._features_rest = wm_sh_reshaped

        # Render
        render_pkg = render(data['camera'], gaussians, data['time'], bg, stage="fine")
        rendered_image = render_pkg["render"].unsqueeze(0)

        # *恢复*: 保持对象状态整洁
        gaussians._features_rest = original_handle

        # --- Step 3: Loss Calculation ---

        # 3.1 Visual Loss (渲染图要像原图)
        l1_vis = l1_loss(rendered_image, gt_image)
        loss_vis = (1.0 - opt.lambda_dssim) * l1_vis + opt.lambda_dssim * (1.0 - ssim(rendered_image, gt_image))

        # 3.2 Watermark Extraction Loss (从参数中提取)
        _, w_logits = inn_model.extract_and_restore(wm_sh_flat, gt_watermark=watermark_key)
        loss_ber = compute_ber_loss(w_logits, watermark_key)

        total_loss = loss_vis + 0.1 * loss_ber

        total_loss.backward()
        inn_optimizer.step()

        # --- Logging ---
        with torch.no_grad():
            cur_psnr = psnr(rendered_image, gt_image).mean().double().item()
            cur_acc = compute_accuracy(w_logits, watermark_key)
            ema_psnr = 0.4 * cur_psnr + 0.6 * ema_psnr
            ema_acc = 0.4 * cur_acc + 0.6 * ema_acc

            if iteration % 100 == 0:
                progress_bar.set_postfix({"PSNR": f"{ema_psnr:.2f}", "Acc": f"{ema_acc:.2f}"})

    # ==========================================================================
    # 保存结果
    # ==========================================================================
    print(f"\n[INFO] Saving results to {opt.workspace}...")
    os.makedirs(opt.workspace, exist_ok=True)

    # 1. 生成最终的带水印参数并替换 GS 模型
    with torch.no_grad():
        final_wm_sh = inn_model.embed(clean_sh_flat, watermark_key).view(N_points, 15, 3)
        # 永久替换以便保存 ply
        gaussians._features_rest = nn.Parameter(final_wm_sh)
        gaussians.save_ply(os.path.join(opt.workspace, "point_cloud_watermarked.ply"))

    # 2. 保存 INN 模型和 Key (用于恢复)
    torch.save({
        'state_dict': inn_model.state_dict(),
        'wm_len': opt.wm_len,
        'alpha': inn_model.alpha
    }, os.path.join(opt.workspace, "inn_model.pth"))

    torch.save(watermark_key, os.path.join(opt.workspace, "watermark_key.pth"))
    print("[INFO] Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 基本路径参数
    parser.add_argument('path', type=str, help="Dataset path")
    parser.add_argument('--pretrained_model_path', type=str, required=True)
    parser.add_argument('--workspace', type=str, default='output/param_watermark')
    parser.add_argument('--watermark_iters', type=int, default=2000)
    parser.add_argument('--wm_len', type=int, default=64)
    parser.add_argument('--lambda_dssim', type=float, default=0.2)
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1])

    # --- 修复部分：补充 GaussianModel 必须的默认参数 ---
    # Model Params
    parser.add_argument('--sh_degree', type=int, default=3)
    parser.add_argument('--source_path', type=str, default="")
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--images', type=str, default="images")
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--white_background', action='store_true')
    parser.add_argument('--data_device', type=str, default="cuda")
    parser.add_argument('--eval', action='store_true')

    # Optimization Params (这是报错缺失的部分)
    parser.add_argument('--position_lr_init', type=float, default=0.00016)
    parser.add_argument('--position_lr_final', type=float, default=0.0000016)
    parser.add_argument('--position_lr_delay_mult', type=float, default=0.01)
    parser.add_argument('--position_lr_max_steps', type=int, default=30_000)
    parser.add_argument('--feature_lr', type=float, default=0.0025)
    parser.add_argument('--opacity_lr', type=float, default=0.05)
    parser.add_argument('--scaling_lr', type=float, default=0.005)
    parser.add_argument('--rotation_lr', type=float, default=0.001)
    parser.add_argument('--percent_dense', type=float, default=0.01)
    parser.add_argument('--densification_interval', type=int, default=100)
    parser.add_argument('--opacity_reset_interval', type=int, default=3000)
    parser.add_argument('--densify_from_iter', type=int, default=500)
    parser.add_argument('--densify_until_iter', type=int, default=15000)
    parser.add_argument('--densify_grad_threshold', type=float, default=0.0002)
    # -----------------------------------------------------

    opt, _ = parser.parse_known_args()

    # 确保输出目录存在
    opt.model_path = opt.workspace

    seed_everything(0)
    gaussians = GaussianModel(opt)
    device = torch.device('cuda')
    dataset = EndoDataset(opt, device=device, type='train')

    train_param_watermark(opt, dataset.dataloader(), gaussians)