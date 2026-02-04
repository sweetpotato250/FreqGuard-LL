import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
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

# [新增] 引入攻击库
from robustness_utils import RobustnessAttacker


# ==============================================================================
# SECTION 1: INN Class Definitions (保持不变，确保加载权重正确)
# ==============================================================================
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2;
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2];
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2];
        x4 = x02[:, :, :, 1::2]
        return torch.cat([x1 + x2 + x3 + x4, -x1 - x2 + x3 + x4, -x1 + x2 - x3 + x4, x1 - x2 - x3 + x4], dim=1)


class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        in_batch, in_channel, in_height, in_width = x.size()
        out_channel, out_height, out_width = in_channel // 4, 2 * in_height, 2 * in_width
        x1 = x[:, 0:out_channel, :, :] / 2;
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2;
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
        h = torch.zeros([in_batch, out_channel, out_height, out_width]).float().to(x.device)
        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4;
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4;
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
        return h


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
        self.wm_projector = nn.Sequential(nn.Linear(wm_len, self.inn_channels), nn.ReLU(),
                                          nn.Linear(self.inn_channels, self.inn_channels))
        self.wm_extractor = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(self.inn_channels, 128),
                                          nn.ReLU(), nn.Linear(128, wm_len))

    def get_wm_feature(self, watermark, dims):
        return self.wm_projector(watermark).unsqueeze(-1).unsqueeze(-1).expand(*dims)

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

    def extract(self, watermarked_image, gt_watermark=None):
        coeffs_wm = self.dwt(watermarked_image)
        z_active, z_passive = self.split_features(coeffs_wm)

        z_rec = self.inn_forward(z_active)
        w_logits = self.wm_extractor(z_rec)

        # 如果提供了GT，用于训练时的反向计算；测试时 w_to_subtract 仅用于恢复图像
        w_to_subtract = gt_watermark if gt_watermark is not None else (torch.sigmoid(w_logits) > 0.5).float()

        z_clean_active = z_rec - self.alpha * self.get_wm_feature(w_to_subtract, z_rec.shape)

        x_active_clean = self.inn_inverse(z_clean_active)
        coeffs_clean = self.merge_features(x_active_clean, z_passive, coeffs_wm.shape)

        return self.idwt(coeffs_clean), w_logits


def compute_accuracy(w_pred_logits, w_gt):
    return ((torch.sigmoid(w_pred_logits) > 0.5).float() == w_gt).float().mean().item()


def tensor2numpy(tensor):
    img = tensor.squeeze(0).cpu().clamp(0, 1).numpy()
    return (np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8)


# ==============================================================================
# SECTION 2: Robustness Test Logic
# ==============================================================================

def test_watermark(opt):
    # 路径处理
    if not opt.model_path.endswith("/"): opt.model_path += "/"
    output_dir = opt.output_path if opt.output_path else opt.model_path

    # 文件夹准备
    render_dir = os.path.join(output_dir, "renders_wm")
    vis_root = os.path.join(output_dir, "visualizations")  # 用于存放攻击后的效果图

    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(vis_root, exist_ok=True)

    # 1. 加载模型
    print(f"[INFO] Loading Watermarked Model: {opt.model_path}")
    wm_gaussians = GaussianModel(opt)
    wm_gaussians.load_ply(os.path.join(opt.model_path, "point_cloud.ply"))
    wm_gaussians.load_model(opt.model_path)

    print(f"[INFO] Loading Reference Model: {opt.source_model_path}")
    ref_gaussians = GaussianModel(opt)
    ref_gaussians.load_ply(os.path.join(opt.source_model_path, "point_cloud.ply"))
    ref_gaussians.load_model(opt.source_model_path)

    # 2. 加载 INN 和 密钥
    inn_model = WatermarkINN(wm_len=opt.wm_len, alpha=0.1, subbands=opt.subbands).cuda()
    inn_model.load_state_dict(torch.load(os.path.join(opt.model_path, "watermark_inn.pth")))
    inn_model.eval()

    watermark_key = torch.load(os.path.join(opt.model_path, "watermark_key.pth")).cuda()

    loss_fn_lpips = lpips.LPIPS(net='alex').cuda()
    device = torch.device('cuda')
    dataset = EndoDataset(opt, device=device, type='test')
    dataloader = dataset.dataloader()

    # --- 定义所有攻击类型 ---
    # 使用 lambda 包装 RobustnessAttacker 的静态方法
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

    # 筛选特定攻击（如果用户指定）
    if opt.specific_attack != 'all':
        if opt.specific_attack in attack_dict:
            attack_dict = {opt.specific_attack: attack_dict[opt.specific_attack]}
        else:
            print(f"[WARN] Attack {opt.specific_attack} not found, running all.")

    # 预创建可视化子文件夹
    for atk_name in attack_dict.keys():
        os.makedirs(os.path.join(vis_root, atk_name), exist_ok=True)

    # 初始化指标容器
    metrics = {'psnr': [], 'ssim': [], 'lpips': []}
    for atk_name in attack_dict.keys():
        metrics[f'acc_{atk_name}'] = []

    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    report_path = os.path.join(output_dir, "robustness_report.txt")

    print(f"[INFO] Testing started.")
    print(f"[INFO] Report file: {report_path}")
    print(f"[INFO] Visualizations: {vis_root}/")

    # 打开报告文件准备写入
    with open(report_path, "w") as f:
        # 写入元数据
        f.write(f"Reference Model: {opt.source_model_path}\n")
        f.write(f"Watermarked Model: {opt.model_path}\n")
        f.write(f"Subbands: {opt.subbands}\n")
        f.write("-" * 80 + "\n")

        # 动态构建表头 (ID | Quality Metrics | Attack Accuracies)
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

                # --- 准备 Mask ---
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

                # --- 渲染图像 ---
                # 1. 参考图 (Ref) - 用于计算视觉质量 (PSNR/SSIM)
                render_pkg_ref = render(data['camera'], ref_gaussians, data['time'], bg, stage="fine")
                ref_image = render_pkg_ref["render"].unsqueeze(0)

                # 2. 水印图 (WM) - 攻击和提取的基础
                render_pkg_wm = render(data['camera'], wm_gaussians, data['time'], bg, stage="fine")
                wm_image = render_pkg_wm["render"].unsqueeze(0)

                # --- Step 1: 视觉质量指标 (Visual Quality) ---
                # 评估“水印对画质的影响”，这一步不受攻击影响，对比的是 WM 和 Ref
                cur_psnr = psnr(wm_image * tool_mask, ref_image * tool_mask).mean().double().item()
                cur_ssim = ssim(wm_image * tool_mask, ref_image * tool_mask).mean().item()
                cur_lpips = loss_fn_lpips(torch.clamp(wm_image, 0, 1) * 2 - 1,
                                          torch.clamp(ref_image, 0, 1) * 2 - 1).mean().item()

                metrics['psnr'].append(cur_psnr)
                metrics['ssim'].append(cur_ssim)
                metrics['lpips'].append(cur_lpips)

                # 记录这一行的基础信息
                row_str = f"{name:<6} | {cur_psnr:<6.2f} | {cur_ssim:<6.3f} | {cur_lpips:<6.3f}"

                # --- Step 2: 鲁棒性测试循环 (Attack & Extract) ---
                for atk_name, attack_fn in attack_dict.items():

                    # [关键点] 1. 施加攻击
                    # 注意：必须 clone()，否则后续攻击会在前一个攻击的基础上叠加
                    attacked_image = attack_fn(wm_image.clone())

                    # [关键点] 2. 可视化保存 (保存前20张)
                    if i < 20:
                        save_path = os.path.join(vis_root, atk_name, f"{name}.png")
                        torchvision.utils.save_image(attacked_image, save_path)

                    # [关键点] 3. 准备提取输入
                    # 模拟接收端行为：只知道要在这个区域提取水印。
                    # 注意：对于几何攻击(Rot/Crop)，直接乘原 Mask 会导致不对齐，Acc 会下降，这是预期的。
                    extract_input = attacked_image * final_embed_mask

                    # [关键点] 4. 提取水印 & 计算 Bit Error
                    _, w_logits = inn_model.extract(extract_input, gt_watermark=None)
                    cur_acc = compute_accuracy(w_logits, watermark_key)

                    # 记录数据
                    metrics[f'acc_{atk_name}'].append(cur_acc)
                    row_str += f" | {cur_acc:<6.2f}"

                # 写入该帧结果
                f.write(row_str + "\n")

        # --- Summary (汇总平均值) ---
        f.write("-" * len(header) + "\n")

        avg_str = f"{'AVG':<6} | {np.mean(metrics['psnr']):<6.2f} | {np.mean(metrics['ssim']):<6.3f} | {np.mean(metrics['lpips']):<6.3f}"

        for atk_name in attack_dict.keys():
            avg_acc = np.mean(metrics[f'acc_{atk_name}'])
            avg_str += f" | {avg_acc:<6.2f}"

        f.write(avg_str + "\n")

    print(f"\n[DONE] Test Finished.")
    print(f"Summary saved to {report_path}")
    print(f"Average PSNR: {np.mean(metrics['psnr']):.2f}")
    print(f"Average Clean Acc: {np.mean(metrics['acc_Clean']):.2f}")
    if 'Noise' in metrics:
        print(f"Average Noise Acc: {np.mean(metrics['acc_Noise']):.2f}")
    if 'JPEG' in metrics:
        print(f"Average JPEG Acc:  {np.mean(metrics['acc_JPEG']):.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Dataset path")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to Watermarked Model (folder containing point_cloud.ply)")
    parser.add_argument('--source_model_path', type=str, required=True,
                        help="Path to Clean Pretrained Model (Reference)")
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--wm_len', type=int, default=64)

    # 子带参数必须与训练时一致，否则 INN 结构不匹配会报错
    parser.add_argument('--subbands', type=str, default='all',
                        help='MUST match the training setting. e.g. all, LL, HL, LH, HH')

    # 调试选项：只运行特定攻击
    parser.add_argument('--specific_attack', type=str, default='all',
                        help='Options: all, Clean, Noise, Rot, Scale, Blur, Crop, Bright, JPEG, Comb')

    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1])

    # GS Params (保持默认即可)
    parser.add_argument('--sh_degree', type=int, default=3)
    parser.add_argument('--percent_dense', type=float, default=0.01)

    opt, _ = parser.parse_known_args()
    seed_everything(0)
    test_watermark(opt)