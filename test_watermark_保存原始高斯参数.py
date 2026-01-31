import torch
import torch.nn as nn
import os
import argparse
import numpy as np
from tqdm import tqdm
import imageio

from gaussian_core.provider import EndoDataset
from gaussian_core.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.image_utils import psnr

# ==============================================================================
# SECTION 1: 将 INN 类直接包含在此处，避免 import 报错
# ==============================================================================

class PointCouplingLayer(nn.Module):
    def __init__(self, in_channels, mid_channels=64):
        super(PointCouplingLayer, self).__init__()
        self.split_len1 = in_channels // 2
        self.split_len2 = in_channels - self.split_len1

        self.net = nn.Sequential(
            nn.Conv1d(self.split_len1, mid_channels, 1),
            nn.ReLU(),
            nn.Conv1d(mid_channels, mid_channels, 1),
            nn.ReLU(),
            nn.Conv1d(mid_channels, self.split_len2 * 2, 1)
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

class ParameterINN(nn.Module):
    def __init__(self, in_channels=45, steps=4, wm_len=64, alpha=0.1):
        super(ParameterINN, self).__init__()
        self.layers = nn.ModuleList([PointCouplingLayer(in_channels) for _ in range(steps)])
        self.wm_len = wm_len
        self.alpha = alpha

        self.wm_projector = nn.Sequential(
            nn.Linear(wm_len, 128),
            nn.ReLU(),
            nn.Linear(128, in_channels)
        )

        self.wm_extractor = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, wm_len)
        )

    def get_wm_feature(self, watermark, n_points):
        feat = self.wm_projector(watermark).unsqueeze(-1)
        return feat.expand(1, -1, n_points)

    def forward_inn(self, x):
        for layer in self.layers:
            x = layer(x, reverse=False)
        return x

    def inverse_inn(self, z):
        for layer in reversed(self.layers):
            z = layer(z, reverse=True)
        return z

    def extract_and_restore(self, params_wm, gt_watermark=None):
        x_wm = params_wm.unsqueeze(0).permute(0, 2, 1)
        N = x_wm.shape[2]

        # 1. Forward to Latent
        z_wm = self.forward_inn(x_wm)

        # 2. Extract Watermark
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

def compute_accuracy(w_pred_logits, w_gt):
    return ((torch.sigmoid(w_pred_logits) > 0.5).float() == w_gt).float().mean().item()

def tensor2numpy(tensor):
    img = tensor.squeeze(0).cpu().clamp(0, 1).numpy()
    return (np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8)


# ==============================================================================
# SECTION 2: 测试逻辑
# ==============================================================================

def test_param_recovery(opt):
    print(f"[INFO] Testing Parameter Restoration...")

    # 1. 加载带水印的 Gaussian Model
    gaussians = GaussianModel(opt)
    gaussians.load_ply(os.path.join(opt.model_path, "point_cloud_watermarked.ply"))

    # 2. 加载 INN 和 Key
    ckpt = torch.load(os.path.join(opt.model_path, "inn_model.pth"))
    inn_model = ParameterINN(in_channels=45, wm_len=ckpt['wm_len'], alpha=ckpt['alpha']).cuda()
    inn_model.load_state_dict(ckpt['state_dict'])
    inn_model.eval()

    key = torch.load(os.path.join(opt.model_path, "watermark_key.pth")).cuda()

    # 3. 执行参数复原 (Parameter Restoration)
    with torch.no_grad():
        # 获取带水印的参数
        wm_sh = gaussians._features_rest  # [N, 15, 3]
        N = wm_sh.shape[0]
        wm_sh_flat = wm_sh.view(N, -1)

        print(f"[INFO] Restoring parameters for {N} points...")

        # INN Extract & Restore
        clean_sh_flat_restored, w_logits = inn_model.extract_and_restore(wm_sh_flat, gt_watermark=key)

        acc = compute_accuracy(w_logits, key)
        print(f"[INFO] Watermark Detection Accuracy from Parameters: {acc * 100:.2f}%")

        # 将复原的参数 reshape 回去
        clean_sh_restored = clean_sh_flat_restored.view(N, 15, 3)

    # 4. 渲染对比
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    dataset = EndoDataset(opt, device=torch.device('cuda'), type='test')
    dataloader = dataset.dataloader()

    save_dir = os.path.join(opt.model_path, "test_vis")
    os.makedirs(save_dir, exist_ok=True)

    psnr_wm_list = []
    psnr_rec_list = []

    for i, data in enumerate(tqdm(dataloader, desc="Rendering")):
        gt = data['camera'].original_image.cuda().unsqueeze(0)

        # --- A. Render Watermarked ---
        pkg_wm = render(data['camera'], gaussians, data['time'], bg, stage="fine")
        img_wm = pkg_wm["render"].unsqueeze(0)

        # --- B. Render Restored ---
        original_handle = gaussians._features_rest
        gaussians._features_rest = clean_sh_restored  # 临时替换

        pkg_rec = render(data['camera'], gaussians, data['time'], bg, stage="fine")
        img_rec = pkg_rec["render"].unsqueeze(0)

        # 恢复指针
        gaussians._features_rest = original_handle

        # --- Metrics ---
        p_wm = psnr(img_wm, gt).mean().item()
        p_rec = psnr(img_rec, gt).mean().item()

        psnr_wm_list.append(p_wm)
        psnr_rec_list.append(p_rec)

        # Save comparison
        if i % 10 == 0:
            combined = torch.cat([gt, img_wm, img_rec], dim=3)  # [GT, WM, Restored]
            imageio.imwrite(os.path.join(save_dir, f"compare_{i:04d}.png"), tensor2numpy(combined))

    print(f"\n[RESULT]")
    print(f"Avg PSNR (Watermarked): {np.mean(psnr_wm_list):.2f}")
    print(f"Avg PSNR (Restored):    {np.mean(psnr_rec_list):.2f}")
    print(f"Note: Restored PSNR should be extremely close to the original model's performance.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Dataset path")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Folder containing point_cloud_watermarked.ply and inn_model.pth")
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1])

    # Model Params (Standard 3DGS arguments)
    parser.add_argument('--sh_degree', type=int, default=3)
    parser.add_argument('--source_path', type=str, default="")
    parser.add_argument('--images', type=str, default="images")
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--white_background', action='store_true')
    parser.add_argument('--data_device', type=str, default="cuda")
    parser.add_argument('--eval', action='store_true')

    # Optimization Params (解决 AttributeError 的关键)
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

    opt, _ = parser.parse_known_args()

    test_param_recovery(opt)