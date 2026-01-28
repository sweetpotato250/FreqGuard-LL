import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from tqdm import tqdm
import numpy as np
import torchvision  # [新增] 用于保存图片

from gaussian_core.provider import EndoDataset
from gaussian_core.utils import seed_everything
from gaussian_core.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.loss_utils import ssim
from utils.image_utils import psnr


# ... [保留之前的 DWT, IDWT, CouplingLayer, WatermarkINN 类定义，不需要变] ...
# ... [为了节省篇幅，这里省略网络定义部分，请保持你之前代码里的类定义不变] ...
# (请确保 DWT, IDWT, CouplingLayer, WatermarkINN, compute_accuracy 都在这里)

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

    def extract(self, watermarked_image):
        coeffs = self.dwt(watermarked_image)
        x = coeffs
        for layer in reversed(self.layers):
            x = layer(x, reverse=True)
        w_pred_logits = self.wm_extract_head(x)
        restored_image = self.idwt(x)
        return restored_image, w_pred_logits


def compute_accuracy(w_pred_logits, w_gt):
    w_pred_bits = (torch.sigmoid(w_pred_logits) > 0.5).float()
    correct_bits = (w_pred_bits == w_gt).float()
    return correct_bits.mean().item()


def test_watermark(opt):
    model_path = opt.model_path
    if not model_path.endswith("/"): model_path += "/"

    if opt.output_path is not None:
        output_dir = opt.output_path
    else:
        output_dir = model_path
    os.makedirs(output_dir, exist_ok=True)

    # [新增] 如果需要保存图片，创建 renders 文件夹
    render_dir = os.path.join(output_dir, "renders")
    if opt.save_images:
        os.makedirs(render_dir, exist_ok=True)
        print(f"[INFO] Rendered images will be saved to: {render_dir}")

    print(f"[INFO] Loading Watermarked Gaussians from: {model_path}")

    gaussians = GaussianModel(opt)
    gaussians.load_ply(os.path.join(model_path, "point_cloud.ply"))
    gaussians.load_model(model_path)

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print(f"[INFO] Loading INN Decoder and Key...")
    inn_model = WatermarkINN().cuda()
    inn_ckpt_path = os.path.join(model_path, "watermark_inn.pth")
    key_ckpt_path = os.path.join(model_path, "watermark_key.pth")

    if not os.path.exists(inn_ckpt_path):
        raise FileNotFoundError(f"Missing watermark_inn.pth in {model_path}")
    if not os.path.exists(key_ckpt_path):
        raise FileNotFoundError(f"Missing watermark_key.pth in {model_path}")

    inn_model.load_state_dict(torch.load(inn_ckpt_path))
    inn_model.eval()
    watermark_key = torch.load(key_ckpt_path).cuda()

    device = torch.device('cuda')
    dataset = EndoDataset(opt, device=device, type='test')
    dataloader = dataset.dataloader()

    output_file = os.path.join(output_dir, "test_metrics_report.txt")

    total_psnr, total_ssim, total_acc = 0.0, 0.0, 0.0
    count = 0

    with open(output_file, "w") as f:
        header = f"{'Image_ID':<15} | {'PSNR':<10} | {'SSIM':<10} | {'Accuracy':<10}\n"
        f.write(header)
        f.write("-" * 55 + "\n")

        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader, desc="Testing")):
                gt_image = data['camera'].original_image.cuda().unsqueeze(0)
                mask = data['mask'].cuda().unsqueeze(0).unsqueeze(0)

                # 1. 渲染 (Inference)
                render_pkg = render(data['camera'], gaussians, data['time'], background, stage="fine")
                rendered_image = render_pkg["render"].unsqueeze(0)

                # 2. 提取水印
                _, w_logits = inn_model.extract(rendered_image)

                # 3. 计算指标
                masked_render = rendered_image * mask
                masked_gt = gt_image * mask

                cur_psnr = psnr(masked_render, masked_gt).mean().double().item()
                cur_ssim = ssim(masked_render, masked_gt).mean().item()
                cur_acc = compute_accuracy(w_logits, watermark_key)

                total_psnr += cur_psnr
                total_ssim += cur_ssim
                total_acc += cur_acc
                count += 1

                image_id = f"img_{i:04d}"
                if hasattr(data['camera'], 'image_name'):
                    image_id = data['camera'].image_name

                # [新增] 保存图片
                if opt.save_images:
                    save_path = os.path.join(render_dir, f"{image_id}.png")
                    torchvision.utils.save_image(rendered_image, save_path)

                line = f"{image_id:<15} | {cur_psnr:<10.4f} | {cur_ssim:<10.4f} | {cur_acc:<10.4f}\n"
                f.write(line)

        f.write("-" * 55 + "\n")
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        avg_acc = total_acc / count
        summary = f"{'AVERAGE':<15} | {avg_psnr:<10.4f} | {avg_ssim:<10.4f} | {avg_acc:<10.4f}\n"
        f.write(summary)

    print(f"\n[DONE] Evaluation Complete.")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average Acc : {avg_acc * 100:.2f}%")
    if opt.save_images:
        print(f"Rendered images saved to: {render_dir}")
    print(f"Full report saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Dataset path")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained watermark model folder")
    parser.add_argument('--output_path', type=str, default=None, help="Optional: Path to save the test report")
    parser.add_argument('--save_images', action='store_true', help="If set, save rendered images to disk")  # [新增开关]
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1])

    # 默认参数
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

    test_watermark(opt)