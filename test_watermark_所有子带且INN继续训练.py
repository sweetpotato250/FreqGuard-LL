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


# ==============================================================================
# SECTION 1: INN 类定义 (必须与训练代码完全一致)
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

    def extract(self, watermarked_image, gt_watermark=None):
        # 1. DWT
        coeffs_wm = self.dwt(watermarked_image)
        # 2. INN Forward -> Latent
        z_rec = self.inn_forward(coeffs_wm)
        # 3. Extract Watermark
        w_logits = self.wm_extractor(z_rec)

        # 4. Strip Watermark (Blindly if gt_watermark is None)
        if gt_watermark is not None:
            w_to_subtract = gt_watermark
        else:
            w_to_subtract = (torch.sigmoid(w_logits) > 0.5).float()

        z_clean = z_rec - self.alpha * self.get_wm_feature(w_to_subtract, z_rec.shape)

        # 5. Restore
        return self.idwt(self.inn_inverse(z_clean)), w_logits


def compute_accuracy(w_pred_logits, w_gt):
    return ((torch.sigmoid(w_pred_logits) > 0.5).float() == w_gt).float().mean().item()


def tensor2numpy(tensor):
    img = tensor.squeeze(0).cpu().clamp(0, 1).numpy()
    return (np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8)


# ==============================================================================
# SECTION 2: 测试逻辑
# ==============================================================================

def test_watermark(opt):
    # 路径准备
    if not opt.model_path.endswith("/"): opt.model_path += "/"
    output_dir = opt.output_path if opt.output_path else opt.model_path
    render_dir = os.path.join(output_dir, "renders")
    gt_dir = os.path.join(output_dir, "gt")
    restored_dir = os.path.join(output_dir, "restored")
    for d in [render_dir, gt_dir, restored_dir]: os.makedirs(d, exist_ok=True)

    print(f"[INFO] Loading Model: {opt.model_path}")

    # 1. Load GS
    gaussians = GaussianModel(opt)
    gaussians.load_ply(os.path.join(opt.model_path, "point_cloud.ply"))
    gaussians.load_model(opt.model_path)

    # 2. Load INN & Key
    inn_model = WatermarkINN(wm_len=opt.wm_len, alpha=0.1).cuda()
    inn_model.load_state_dict(torch.load(os.path.join(opt.model_path, "watermark_inn.pth")))
    inn_model.eval()

    watermark_key = torch.load(os.path.join(opt.model_path, "watermark_key.pth")).cuda()
    print(f"[INFO] Key Loaded. First 10 bits: {watermark_key[0, :10].cpu().numpy()}")

    # 3. Metrics & Data
    loss_fn_lpips = lpips.LPIPS(net='alex').cuda()
    device = torch.device('cuda')
    dataset = EndoDataset(opt, device=device, type='test')
    dataloader = dataset.dataloader()

    video_writer = imageio.get_writer(os.path.join(output_dir, "comparison.mp4"), fps=24, macro_block_size=1)

    metrics = {'psnr': [], 'ssim': [], 'lpips': [], 'acc': [], 'rec_psnr': []}

    with open(os.path.join(output_dir, "report.txt"), "w") as f:
        f.write(f"{'ID':<10} | {'PSNR':<8} | {'SSIM':<8} | {'LPIPS':<8} | {'ACC':<8} | {'REC_PSNR':<8}\n")
        f.write("-" * 70 + "\n")

        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader, desc="Testing")):
                gt = data['camera'].original_image.cuda().unsqueeze(0)
                mask = data['mask'].cuda().unsqueeze(0).unsqueeze(0)
                bg = torch.zeros(3, dtype=torch.float32, device="cuda")

                # Render
                render_pkg = render(data['camera'], gaussians, data['time'], bg, stage="fine")
                rendered = render_pkg["render"].unsqueeze(0)

                # Extract & Restore (Blind)
                restored, w_logits = inn_model.extract(rendered, gt_watermark=None)

                # Masking
                m_render, m_gt, m_restore = rendered * mask, gt * mask, restored * mask

                # Metrics
                cur_psnr = psnr(m_render, m_gt).mean().double().item()
                cur_ssim = ssim(m_render, m_gt).mean().item()
                cur_lpips = loss_fn_lpips(torch.clamp(m_render, 0, 1) * 2 - 1,
                                          torch.clamp(m_gt, 0, 1) * 2 - 1).mean().item()
                cur_acc = compute_accuracy(w_logits, watermark_key)

                # Recovery Metric
                cur_rec = psnr(m_restore, m_gt).mean().double().item()

                metrics['psnr'].append(cur_psnr)
                metrics['ssim'].append(cur_ssim)
                metrics['lpips'].append(cur_lpips)
                metrics['acc'].append(cur_acc)
                metrics['rec_psnr'].append(cur_rec)

                name = data['camera'].image_name if hasattr(data['camera'], 'image_name') else f"{i:04d}"
                f.write(
                    f"{name:<10} | {cur_psnr:<8.2f} | {cur_ssim:<8.4f} | {cur_lpips:<8.4f} | {cur_acc:<8.4f} | {cur_rec:<8.2f}\n")

                # Save Images
                torchvision.utils.save_image(rendered, os.path.join(render_dir, f"{name}.png"))
                torchvision.utils.save_image(gt, os.path.join(gt_dir, f"{name}.png"))
                torchvision.utils.save_image(restored, os.path.join(restored_dir, f"{name}.png"))

                # Video Frame: GT | Render | Restored
                combined = torch.cat([gt, rendered, restored], dim=3)
                video_writer.append_data(tensor2numpy(combined))

        avg = {k: np.mean(v) for k, v in metrics.items()}
        f.write("-" * 70 + "\n")
        f.write(
            f"{'AVG':<10} | {avg['psnr']:<8.2f} | {avg['ssim']:<8.4f} | {avg['lpips']:<8.4f} | {avg['acc']:<8.4f} | {avg['rec_psnr']:<8.2f}\n")

    video_writer.close()
    print(f"\n[DONE] Avg Rec_PSNR: {avg['rec_psnr']:.2f} | Avg Acc: {avg['acc']:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Dataset path")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--wm_len', type=int, default=64)
    # [修复点] 显式添加 data_range 参数
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1], help="Data range to use")

    # GS Params (Dummy to avoid errors)
    parser.add_argument('--sh_degree', type=int, default=3)
    parser.add_argument('--percent_dense', type=float, default=0.01)

    opt, _ = parser.parse_known_args()
    seed_everything(0)
    test_watermark(opt)