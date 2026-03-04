import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import math  # [新增]
from tqdm import tqdm
import torchvision
import numpy as np
import imageio
import lpips

from gaussian_core.provider import EndoDataset
from gaussian_core.utils import seed_everything
from gaussian_core.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.loss_utils import ssim
from utils.image_utils import psnr

# 引入攻击库
from robustness_utils import RobustnessAttacker


# ==============================================================================
# SECTION 1: INN Class Definitions
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
        return torch.cat([x1 + x2 + x3 + x4, -x1 - x2 + x3 + x4, -x1 + x2 - x3 + x4, x1 - x2 - x3 + x4], dim=1)


class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        in_batch, in_channel, in_height, in_width = x.size()
        out_channel, out_height, out_width = in_channel // 4, 2 * in_height, 2 * in_width
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


# [第二步对齐] 重要性映射模块
class ImportanceMapModule(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ImportanceMapModule, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class CouplingLayer(nn.Module):
    def __init__(self, in_channels, mid_channels=64):
        super(CouplingLayer, self).__init__()
        self.split_len1 = in_channels // 2
        self.split_len2 = in_channels - self.split_len1
        self.net = nn.Sequential(
            nn.Conv2d(self.split_len1, mid_channels, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, 1, 1, 0), nn.ReLU(),
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

        # [第二步对齐]
        self.im_module = ImportanceMapModule(in_channels=3)

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

        # --- [核心修改] 动态自适应网络容量 (与训练对齐) ---
        target_features = max(128, wm_len * 2)
        self.spatial_size = max(1, math.ceil(math.sqrt(target_features / self.inn_channels)))
        spatial_dim = self.inn_channels * self.spatial_size * self.spatial_size
        hidden_dim = max(256, wm_len * 2)

        self.layers = nn.ModuleList([CouplingLayer(self.inn_channels) for _ in range(steps)])

        self.wm_projector = nn.Sequential(
            nn.Linear(wm_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, spatial_dim)
        )
        self.wm_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.spatial_size, self.spatial_size)),
            nn.Flatten(),
            nn.Linear(spatial_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, wm_len)
        )

    def get_wm_feature(self, watermark, dims):
        B, C, H, W = dims
        feat = self.wm_projector(watermark)
        feat = feat.view(B, self.inn_channels, self.spatial_size, self.spatial_size)
        return F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)

    def inn_forward(self, x):
        for layer in self.layers: x = layer(x, reverse=False)
        return x

    def inn_inverse(self, z):
        for layer in reversed(self.layers): z = layer(z, reverse=True)
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

    # 测试阶段仅调用 extract，但保留 embed 结构完整性
    def embed(self, image, watermark):
        coeffs = self.dwt(image)
        ll_subband = coeffs[:, :3, :, :]
        M = self.im_module(ll_subband)

        z_active, z_passive = self.split_features(coeffs)
        z_transformed = self.inn_forward(z_active)
        wm_feat = self.get_wm_feature(watermark, z_transformed.shape)

        z_watermarked = z_transformed + self.alpha * M * wm_feat

        x_active_restored = self.inn_inverse(z_watermarked)
        coeffs_watermarked = self.merge_features(x_active_restored, z_passive, coeffs.shape)
        return self.idwt(coeffs_watermarked)

    def extract(self, watermarked_image, gt_watermark=None):
        coeffs_wm = self.dwt(watermarked_image)
        ll_subband_wm = coeffs_wm[:, :3, :, :]
        M = self.im_module(ll_subband_wm)

        z_active, z_passive = self.split_features(coeffs_wm)

        z_rec = self.inn_forward(z_active)
        w_logits = self.wm_extractor(z_rec)

        w_to_subtract = gt_watermark if gt_watermark is not None else (torch.sigmoid(w_logits) > 0.5).float()
        wm_feat_sub = self.get_wm_feature(w_to_subtract, z_rec.shape)

        # [第二步对齐] 应用 Importance Map 进行逆向相减
        z_clean_active = z_rec - self.alpha * M * wm_feat_sub

        x_active_clean = self.inn_inverse(z_clean_active)
        coeffs_clean = self.merge_features(x_active_clean, z_passive, coeffs_wm.shape)

        return self.idwt(coeffs_clean), w_logits


def compute_accuracy(w_pred_logits, w_gt):
    return ((torch.sigmoid(w_pred_logits) > 0.5).float() == w_gt).float().mean().item()


# ==============================================================================
# SECTION 2: Robustness Test Logic
# ==============================================================================

def test_watermark(opt):
    if not opt.model_path.endswith("/"): opt.model_path += "/"
    output_dir = opt.output_path if opt.output_path else opt.model_path

    # [清理] 彻底移除了没有实际写入作用的 render_dir 避免产生空文件夹
    vis_root = os.path.join(output_dir, "visualizations")

    print(f"[INFO] Loading Watermarked Model: {opt.model_path}")
    wm_gaussians = GaussianModel(opt)
    wm_gaussians.load_ply(os.path.join(opt.model_path, "point_cloud.ply"))
    wm_gaussians.load_model(opt.model_path)

    print(f"[INFO] Loading Reference Model: {opt.source_model_path}")
    ref_gaussians = GaussianModel(opt)
    ref_gaussians.load_ply(os.path.join(opt.source_model_path, "point_cloud.ply"))
    ref_gaussians.load_model(opt.source_model_path)

    config_path = os.path.join(opt.model_path, "train_config.txt")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":", 1)
                    if k.strip() == "subbands":
                        opt.subbands = v.strip()
                    elif k.strip() == "wm_len":
                        opt.wm_len = int(v.strip())
        print(f"[INFO] Auto-loaded config: subbands={opt.subbands}, wm_len={opt.wm_len}")

    inn_model = WatermarkINN(wm_len=opt.wm_len, alpha=0.1, subbands=opt.subbands).cuda()
    inn_model.load_state_dict(torch.load(os.path.join(opt.model_path, "watermark_inn.pth")))
    inn_model.eval()

    watermark_key = torch.load(os.path.join(opt.model_path, "watermark_key.pth")).cuda()

    loss_fn_lpips = lpips.LPIPS(net='alex').cuda()
    device = torch.device('cuda')
    dataset = EndoDataset(opt, device=device, type='test')
    dataloader = dataset.dataloader()

    attack_dict = {
        "Clean": lambda x: x,
        "Noise": lambda x: RobustnessAttacker.attack_noise(x, std=0.1),
        "Rot": lambda x: RobustnessAttacker.attack_rotation(x, angle_deg=30),
        "Scale": lambda x: RobustnessAttacker.attack_scaling(x, scale=0.25),
        "Blur": lambda x: RobustnessAttacker.attack_blur(x, sigma=1.0),
        "Crop": lambda x: RobustnessAttacker.attack_crop(x, crop_percent=0.4),
        "Bright": lambda x: RobustnessAttacker.attack_brightness(x, factor=2.0),
        "JPEG": lambda x: RobustnessAttacker.attack_jpeg(x, quality=10),
        "Comb": lambda x: RobustnessAttacker.attack_combined(x)
    }

    if opt.specific_attack != 'all':
        if opt.specific_attack in attack_dict:
            attack_dict = {opt.specific_attack: attack_dict[opt.specific_attack]}
        else:
            print(f"[WARN] Attack {opt.specific_attack} not found, running all.")

    # [清理] 移除提前批量创建空文件夹的逻辑，改为稍后在保存时懒加载创建

    metrics = {'psnr': [], 'ssim': [], 'lpips': []}
    for atk_name in attack_dict.keys():
        metrics[f'acc_{atk_name}'] = []

    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    report_path = os.path.join(output_dir, "robustness_report.txt")

    # 确保根目录存在用于存 txt 报告
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Testing started.")

    with open(report_path, "w") as f:
        f.write(f"Reference Model: {opt.source_model_path}\n")
        f.write(f"Watermarked Model: {opt.model_path}\n")
        f.write(f"Subbands: {opt.subbands}\n")
        f.write("-" * 80 + "\n")

        header = f"{'ID':<6} | {'PSNR':<6} | {'SSIM':<6} | {'LPIPS':<6}"
        for atk_name in attack_dict.keys():
            header += f" | {atk_name[:6]:<6}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader, desc="Robustness Testing")):
                name = data['camera'].image_name if hasattr(data['camera'], 'image_name') else f"{i:04d}"

                gt_sensor = data['camera'].original_image.cuda().unsqueeze(0)
                B, C, H, W = gt_sensor.shape

                tool_mask = data['mask'].cuda().unsqueeze(0).unsqueeze(0).float()
                if tool_mask.shape[2] != H or tool_mask.shape[3] != W:
                    tool_mask = F.interpolate(tool_mask, size=(H, W), mode='nearest')

                if 'lesion_mask' in data:
                    lesion_mask = data['lesion_mask'].cuda().unsqueeze(0).unsqueeze(0).float()
                    if lesion_mask.shape[2] != H or lesion_mask.shape[3] != W:
                        lesion_mask = F.interpolate(lesion_mask, size=(H, W), mode='nearest')
                else:
                    lesion_mask = torch.ones_like(tool_mask)

                roni_mask = tool_mask * lesion_mask
                erosion_kernel = 21
                inverted_roni = 1.0 - roni_mask
                dilated_inverted = F.max_pool2d(inverted_roni, kernel_size=erosion_kernel, stride=1,
                                                padding=erosion_kernel // 2)
                final_embed_mask = 1.0 - dilated_inverted

                render_pkg_ref = render(data['camera'], ref_gaussians, data['time'], bg, stage="fine")
                ref_image = render_pkg_ref["render"].unsqueeze(0)

                render_pkg_wm = render(data['camera'], wm_gaussians, data['time'], bg, stage="fine")
                wm_image = render_pkg_wm["render"].unsqueeze(0)

                cur_psnr = psnr(wm_image * tool_mask, ref_image * tool_mask).mean().double().item()
                cur_ssim = ssim(wm_image * tool_mask, ref_image * tool_mask).mean().item()
                cur_lpips = loss_fn_lpips(torch.clamp(wm_image, 0, 1) * 2 - 1,
                                          torch.clamp(ref_image, 0, 1) * 2 - 1).mean().item()

                metrics['psnr'].append(cur_psnr)
                metrics['ssim'].append(cur_ssim)
                metrics['lpips'].append(cur_lpips)

                row_str = f"{name:<6} | {cur_psnr:<6.2f} | {cur_ssim:<6.3f} | {cur_lpips:<6.3f}"

                for atk_name, attack_fn in attack_dict.items():

                    attacked_image = attack_fn(wm_image.clone())

                    if i < 20:
                        # [清理] 只有真正需要存图时，才生成对应的可视化文件夹，彻底杜绝生成空文件夹
                        save_dir = os.path.join(vis_root, atk_name)
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"{name}.png")
                        torchvision.utils.save_image(attacked_image, save_path)

                    extract_input = attacked_image * final_embed_mask

                    _, w_logits = inn_model.extract(extract_input, gt_watermark=None)
                    cur_acc = compute_accuracy(w_logits, watermark_key)

                    metrics[f'acc_{atk_name}'].append(cur_acc)
                    row_str += f" | {cur_acc:<6.2f}"

                f.write(row_str + "\n")

        f.write("-" * len(header) + "\n")
        avg_str = f"{'AVG':<6} | {np.mean(metrics['psnr']):<6.2f} | {np.mean(metrics['ssim']):<6.3f} | {np.mean(metrics['lpips']):<6.3f}"
        for atk_name in attack_dict.keys():
            avg_acc = np.mean(metrics[f'acc_{atk_name}'])
            avg_str += f" | {avg_acc:<6.2f}"

        f.write(avg_str + "\n")

    print(f"\n[DONE] Test Finished.")
    print(f"Summary saved to {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Dataset path")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--source_model_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--wm_len', type=int, default=64)
    parser.add_argument('--subbands', type=str, default='all')
    parser.add_argument('--specific_attack', type=str, default='all')
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1])
    parser.add_argument('--sh_degree', type=int, default=3)
    parser.add_argument('--percent_dense', type=float, default=0.01)

    opt, _ = parser.parse_known_args()
    seed_everything(0)
    test_watermark(opt)