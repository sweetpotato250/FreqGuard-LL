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
# SECTION 1: INN 类定义 (保持一致)
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

    gaussians = GaussianModel(opt)
    gaussians.load_ply(os.path.join(opt.model_path, "point_cloud.ply"))
    gaussians.load_model(opt.model_path)

    inn_model = WatermarkINN(wm_len=opt.wm_len, alpha=0.1).cuda()
    inn_model.load_state_dict(torch.load(os.path.join(opt.model_path, "watermark_inn.pth")))
    inn_model.eval()

    watermark_key = torch.load(os.path.join(opt.model_path, "watermark_key.pth")).cuda()
    print(f"[INFO] Key Loaded. First 10 bits: {watermark_key[0, :10].cpu().numpy()}")

    loss_fn_lpips = lpips.LPIPS(net='alex').cuda()
    device = torch.device('cuda')
    dataset = EndoDataset(opt, device=device, type='test')
    dataloader = dataset.dataloader()

    video_writer = imageio.get_writer(os.path.join(output_dir, "comparison.mp4"), fps=24, macro_block_size=1)

    metrics = {'psnr': [], 'ssim': [], 'lpips': [], 'acc': [], 'rec_psnr': []}
    warned_missing_mask = False

    with open(os.path.join(output_dir, "report.txt"), "w") as f:
        f.write(f"{'ID':<10} | {'PSNR':<8} | {'SSIM':<8} | {'LPIPS':<8} | {'ACC':<8} | {'REC_PSNR':<8}\n")
        f.write("-" * 70 + "\n")

        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader, desc="Testing")):
                gt = data['camera'].original_image.cuda().unsqueeze(0)
                B, C, H, W = gt.shape

                # --- Mask Logic (必须与训练一致) ---
                tool_mask = data['mask'].cuda().unsqueeze(0).unsqueeze(0).float()
                if tool_mask.shape[2] != H or tool_mask.shape[3] != W:
                    tool_mask = F.interpolate(tool_mask, size=(H, W), mode='nearest')

                # 尝试加载 lesion_mask，如果没有则默认全1（只避开器械）
                if 'lesion_mask' in data:
                    lesion_mask = data['lesion_mask'].cuda().unsqueeze(0).unsqueeze(0).float()
                    if lesion_mask.shape[2] != H or lesion_mask.shape[3] != W:
                        lesion_mask = F.interpolate(lesion_mask, size=(H, W), mode='nearest')
                else:
                    if not warned_missing_mask and i == 0:
                        print(f"[WARN] No lesion_mask in test data. Assuming full non-tool area.")
                        warned_missing_mask = True
                    lesion_mask = torch.ones_like(tool_mask)

                roni_mask = tool_mask * lesion_mask
                erosion_kernel = 21
                inverted_roni = 1.0 - roni_mask
                dilated_inverted = F.max_pool2d(inverted_roni, kernel_size=erosion_kernel, stride=1,
                                                padding=erosion_kernel // 2)
                final_embed_mask = 1.0 - dilated_inverted

                bg = torch.zeros(3, dtype=torch.float32, device="cuda")

                # Render
                render_pkg = render(data['camera'], gaussians, data['time'], bg, stage="fine")
                rendered = render_pkg["render"].unsqueeze(0)

                # Extract & Restore (关键修改：应用掩码！)
                # 测试时也必须遮挡器械，否则会提取到器械上的噪声
                extract_input = rendered * final_embed_mask
                restored_masked, w_logits = inn_model.extract(extract_input, gt_watermark=None)

                # 拼接：为了计算 Rec_PSNR，我们需要把复原的背景和原始的器械（或者渲染的器械）拼回去
                # 但更科学的方法是只计算背景的 PSNR。这里我们为了视觉完整性，把 mask 部分用 rendered 填回去
                restored_full = restored_masked * final_embed_mask + rendered * (1.0 - final_embed_mask)

                # Metrics
                # 1. 渲染质量 (Rendered vs GT, masked by Tool)
                cur_psnr = psnr(rendered * tool_mask, gt * tool_mask).mean().double().item()
                cur_ssim = ssim(rendered * tool_mask, gt * tool_mask).mean().item()
                cur_lpips = loss_fn_lpips(torch.clamp(rendered, 0, 1) * 2 - 1,
                                          torch.clamp(gt, 0, 1) * 2 - 1).mean().item()
                cur_acc = compute_accuracy(w_logits, watermark_key)

                # 2. 复原质量 (只看背景)
                cur_rec = psnr(restored_masked * final_embed_mask, gt * final_embed_mask).mean().double().item()

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
                torchvision.utils.save_image(restored_full, os.path.join(restored_dir, f"{name}.png"))

                # Video Frame: GT | Render | Restored
                combined = torch.cat([gt, rendered, restored_full], dim=3)
                video_writer.append_data(tensor2numpy(combined))

        avg = {k: np.mean(v) for k, v in metrics.items()}
        f.write("-" * 70 + "\n")
        f.write(
            f"{'AVG':<10} | {avg['psnr']:<8.2f} | {avg['ssim']:<8.4f} | {avg['lpips']:<8.4f} | {avg['acc']:<8.4f} | {avg['rec_psnr']:<8.2f}\n")

    video_writer.close()
    print(f"\n[DONE] Avg Rec_PSNR (Bg): {avg['rec_psnr']:.2f} | Avg Acc: {avg['acc']:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Dataset path")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--wm_len', type=int, default=64)
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1], help="Data range to use")

    # GS Params (Dummy to avoid errors)
    parser.add_argument('--sh_degree', type=int, default=3)
    parser.add_argument('--percent_dense', type=float, default=0.01)

    opt, _ = parser.parse_known_args()
    seed_everything(0)
    test_watermark(opt)