import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from tqdm import tqdm
import numpy as np

from gaussian_core.provider import EndoDataset
from gaussian_core.utils import *
from gaussian_core.gaussian_model import GaussianModel
from gaussian_renderer import render


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
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / 4), r * in_height, r * in_width
        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

        h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)
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
            # Forward: Additive coupling
            y1 = x1
            y2 = x2 + self.net(x1)
            return torch.cat([y1, y2], dim=1)
        else:
            # Reverse
            y1 = x1
            y2 = x2 - self.net(x1)
            return torch.cat([y1, y2], dim=1)


class WatermarkINN(nn.Module):
    def __init__(self, in_channels=12, steps=3):  # 12 channels because RGB(3) * 4 subbands
        super(WatermarkINN, self).__init__()
        self.dwt = DWT()
        self.idwt = IDWT()

        # INN Layers
        self.layers = nn.ModuleList([CouplingLayer(in_channels) for _ in range(steps)])

        # Watermark Embedder: Projects 64-bit watermark to image feature space
        # We append watermark as a bias to the high-freq channels before INN
        self.wm_len = 64
        self.wm_projection = nn.Linear(self.wm_len, in_channels)

    def embed(self, image, watermark):
        # 1. DWT
        coeffs = self.dwt(image)  # [B, 12, H/2, W/2]

        # 2. Add Watermark Trigger (simple additive for conditioning)
        B, C, H, W = coeffs.shape
        wm_feat = self.wm_projection(watermark).unsqueeze(-1).unsqueeze(-1)  # [B, 12, 1, 1]
        x = coeffs + wm_feat

        # 3. INN Forward
        for layer in self.layers:
            x = layer(x, reverse=False)

        # 4. IDWT
        watermarked_image = self.idwt(x)
        return watermarked_image

    def extract(self, watermarked_image, watermark):
        # 1. DWT
        coeffs = self.dwt(watermarked_image)

        # 2. INN Reverse
        x = coeffs
        for layer in reversed(self.layers):
            x = layer(x, reverse=True)

        # 3. Remove Watermark Trigger to get clean coeffs
        wm_feat = self.wm_projection(watermark).unsqueeze(-1).unsqueeze(-1)
        clean_coeffs = x - wm_feat

        # 4. IDWT to get clean image
        restored_image = self.idwt(clean_coeffs)

        # Note: In a real extraction scenario, we would need a classifier to predict 'watermark' from 'x'.
        # For this copyright protection task, we verify if (Extracted_Image + Key) -> Watermarked_Image matches.
        # Or more simply, we calculate loss on the restored image vs original.
        return restored_image


# ==========================================
# Part 3: Training Loop
# ==========================================
def train_watermark(opt, dataloader, gaussians):
    # 1. Load Pre-trained Model
    print(f"Loading pre-trained model from {opt.pretrained_model_path}")
    gaussians.load_ply(os.path.join(opt.pretrained_model_path, "point_cloud.ply"))
    gaussians.load_model(opt.pretrained_model_path)  # Load deformation

    # 2. Setup Watermark Model
    inn_model = WatermarkINN().cuda()
    optimizer_inn = torch.optim.Adam(inn_model.parameters(), lr=1e-4)

    # 3. Setup Gaussian Optimizer (Smaller LR for fine-tuning)
    gaussians.training_setup()
    for param_group in gaussians.optimizer.param_groups:
        param_group['lr'] *= 0.1  # Reduce LR to preserve quality

    # 4. Define Watermark (Fixed for the model copyright)
    # 64-bit random key
    watermark_key = torch.randint(0, 2, (1, 64)).float().cuda()

    iter_start = 0
    max_iter = opt.watermark_iters
    progress_bar = tqdm(range(iter_start, max_iter), desc="Watermark Embedding")

    bg_color = [0, 0, 0]  # Black background for rendering
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    for iteration in range(iter_start, max_iter):
        # --- Data Loading ---
        try:
            data = next(iter(dataloader))
        except StopIteration:
            dataloader_iter = iter(dataloader)
            data = next(dataloader_iter)

        gt_image = data['camera'].original_image.cuda().unsqueeze(0)  # [1, 3, H, W]
        mask = data['mask'].cuda().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W] (1=Tissue, 0=Tool/Bg)

        # Ensure mask is valid for non-lesion targeting
        # Assuming the mask provided by EndoGS dataloader: 1 is valid tissue area
        valid_mask = mask

        # ==========================================
        # Step A: Train INN (The "Teacher")
        # ==========================================
        # Goal: Learn to embed watermark into GT image invisibly and reversibly
        optimizer_inn.zero_grad()

        # Embed
        wm_image = inn_model.embed(gt_image, watermark_key)

        # Extract (Restore)
        restored_image = inn_model.extract(wm_image, watermark_key)

        # Apply Mask constraint: We only care about embedding in the mask region
        # However, INN works on whole image. We enforce consistency loss heavily on masked region.

        # Loss 1: Imperceptibility (Watermarked vs GT)
        loss_imperceptibility = l1_loss(wm_image * valid_mask, gt_image * valid_mask)

        # Loss 2: Reversibility (Restored vs GT)
        loss_reversibility = l1_loss(restored_image, gt_image)

        # Total INN Loss
        loss_inn = loss_imperceptibility + loss_reversibility
        loss_inn.backward()
        optimizer_inn.step()

        # ==========================================
        # Step B: Fine-tune Gaussians (The "Student")
        # ==========================================
        # Goal: Make Gaussians render the "Watermarked Image" instead of original GT

        gaussians.optimizer.zero_grad(set_to_none=True)

        # Render current view
        render_pkg = render(data['camera'], gaussians, data['time'], background, stage="fine")
        rendered_image = render_pkg["render"].unsqueeze(0)

        # Target is the watermarked image generated by the INN (detached to stop grad to INN)
        target_image = wm_image.detach()

        # Apply Mask: Only calculate loss in the non-lesion/tissue area
        # We want Render -> Watermarked_GT in Tissue area
        # We want Render -> GT in other areas (optional, or just ignore)

        loss_recon = l1_loss(rendered_image * valid_mask, target_image * valid_mask)

        # Optional: Add structural similarity or perceptual loss here if needed

        loss_recon.backward()
        gaussians.optimizer.step()

        # Update progress
        if iteration % 10 == 0:
            progress_bar.set_postfix({
                "INN_Loss": f"{loss_inn.item():.5f}",
                "GS_Loss": f"{loss_recon.item():.5f}"
            })

    # Save Final Watermarked Model
    print(f"Saving watermarked model to {opt.workspace}")
    gaussians.save(opt.workspace, max_iter, "fine")
    torch.save(inn_model.state_dict(), os.path.join(opt.workspace, "watermark_inn.pth"))
    torch.save(watermark_key, os.path.join(opt.workspace, "watermark_key.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Path to dataset")
    parser.add_argument('--pretrained_model_path', type=str, required=True,
                        help="Path to the clean pretrained model (step 1 output)")
    parser.add_argument('--workspace', type=str, default='output/watermarked_model', help="Where to save result")
    parser.add_argument('--watermark_iters', type=int, default=5000, help="Number of fine-tuning iterations")
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1])

    # Default Gaussian params (needed for init)
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

    # Init Gaussian Model
    gaussians = GaussianModel(opt)

    # Init Dataloader
    device = torch.device('cuda')
    dataset = EndoDataset(opt, device=device, type='train')
    dataloader = dataset.dataloader()

    train_watermark(opt, dataloader, gaussians)