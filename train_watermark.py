import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from tqdm import tqdm
import numpy as np

# 修正导入路径，确保能找到这些模块
from gaussian_core.provider import EndoDataset
from gaussian_core.utils import *
from gaussian_core.gaussian_model import GaussianModel
from gaussian_renderer import render
# [FIX] 显式导入 l1_loss
from utils.loss_utils import l1_loss


# ==========================================
# Part 1: DWT & IDWT (Haar Wavelet)
# ==========================================
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        # x: [B, C, H, W]
        # Haar Wavelet Transform
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

        # Output: [B, 4*C, H/2, W/2]
        return torch.cat([x_LL, x_HL, x_LH, x_HH], dim=1)


class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        # x: [B, 4*C, H/2, W/2]
        # Inverse Haar Wavelet Transform
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
        # Split channels
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
    def __init__(self, in_channels=12, steps=3, wm_len=64):
        # in_channels=12 because RGB(3) * 4 subbands = 12
        super(WatermarkINN, self).__init__()
        self.dwt = DWT()
        self.idwt = IDWT()
        self.wm_len = wm_len

        # INN Layers
        self.layers = nn.ModuleList([CouplingLayer(in_channels) for _ in range(steps)])

        # Watermark Projection (Embedding)
        self.wm_projection = nn.Linear(wm_len, in_channels)

        # Watermark Extraction Head (Decoding)
        # Maps feature maps back to 64 bits probability logits
        self.wm_extract_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling [B, C, 1, 1]
            nn.Flatten(),  # [B, C]
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, wm_len)  # Output logits
        )

    def embed(self, image, watermark):
        """
        Embed watermark into image.
        Returns: Watermarked Image (Spatial Domain)
        """
        # 1. DWT: [B, 3, H, W] -> [B, 12, H/2, W/2]
        coeffs = self.dwt(image)

        # 2. Add Watermark Trigger to coefficients
        wm_feat = self.wm_projection(watermark).unsqueeze(-1).unsqueeze(-1)  # [B, 12, 1, 1]
        x = coeffs + wm_feat

        # 3. INN Forward (Mixing)
        for layer in self.layers:
            x = layer(x, reverse=False)

        # 4. IDWT: [B, 12, H/2, W/2] -> [B, 3, H, W]
        watermarked_image = self.idwt(x)
        return watermarked_image

    def extract(self, watermarked_image):
        """
        Extract watermark and restore image.
        Returns: Restored Image, Predicted Watermark Logits
        """
        # 1. DWT
        coeffs = self.dwt(watermarked_image)

        # 2. INN Reverse
        x = coeffs
        for layer in reversed(self.layers):
            x = layer(x, reverse=True)

        # 3. Extract Watermark (Predict logits) from the reversed features
        w_pred_logits = self.wm_extract_head(x)

        # 4. Restore Image (IDWT of reversed features)
        restored_image = self.idwt(x)

        return restored_image, w_pred_logits


def compute_ber_loss(w_pred_logits, w_gt):
    """
    Compute differentiable Bit Error Rate loss (BCEWithLogits).
    """
    # w_gt is 0 or 1, w_pred_logits is real number
    return F.binary_cross_entropy_with_logits(w_pred_logits, w_gt)


def compute_accuracy(w_pred_logits, w_gt):
    """
    Compute exact bit match accuracy (0.0 to 1.0).
    """
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
    gaussians.load_model(opt.pretrained_model_path)  # Load deformation

    # 2. Setup Watermark Model
    inn_model = WatermarkINN().cuda()
    optimizer_inn = torch.optim.Adam(inn_model.parameters(), lr=1e-4)

    # 3. Setup Gaussian Optimizer (Smaller LR for fine-tuning)
    gaussians.training_setup()
    for param_group in gaussians.optimizer.param_groups:
        param_group['lr'] *= 0.1  # Reduce LR to preserve visual quality

    # 4. Define Watermark (Fixed for the model copyright)
    # 64-bit random key (0 or 1)
    watermark_key = torch.randint(0, 2, (1, 64)).float().cuda()
    print(f"Generated Watermark Key: {watermark_key[0, :10]}... (first 10 bits)")

    iter_start = 0
    max_iter = opt.watermark_iters
    progress_bar = tqdm(range(iter_start, max_iter), desc="Watermark Fine-tuning")

    bg_color = [0, 0, 0]  # Black background
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Hyperparameters
    lambda_ber = 0.5  # Weight for BER loss to constrain SH/Opacity
    lambda_w_rec = 1.0  # Weight for Watermark reconstruction in INN training

    for iteration in range(iter_start, max_iter):
        # --- Data Loading ---
        try:
            data = next(iter(dataloader))
        except StopIteration:
            dataloader_iter = iter(dataloader)
            data = next(dataloader_iter)

        gt_image = data['camera'].original_image.cuda().unsqueeze(0)  # [1, 3, H, W]
        mask = data['mask'].cuda().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W] (1=Tissue)

        valid_mask = mask

        # ==========================================
        # Step A: Train INN (The "Teacher")
        # ==========================================
        # Goal: Train INN to embed W into GT invisibly and extract it accurately
        optimizer_inn.zero_grad()

        # Embed
        wm_image = inn_model.embed(gt_image, watermark_key)

        # Extract
        restored_image, w_pred_logits = inn_model.extract(wm_image)

        # Loss 1: Imperceptibility (Watermarked vs GT)
        # We only care about visual quality in the valid tissue area
        loss_imperceptibility = l1_loss(wm_image * valid_mask, gt_image * valid_mask)

        # Loss 2: Watermark Extraction (BCE)
        loss_w_extract = compute_ber_loss(w_pred_logits, watermark_key)

        # Total INN Loss
        loss_inn = loss_imperceptibility + lambda_w_rec * loss_w_extract
        loss_inn.backward()
        optimizer_inn.step()

        # ==========================================
        # Step B: Fine-tune Gaussians (The "Student")
        # ==========================================
        # Goal: Force Gaussians (SH, Opacity, etc.) to render an image that contains the watermark code

        gaussians.optimizer.zero_grad(set_to_none=True)

        # Render current view
        # Gradients will flow from render -> SH/Opacity
        render_pkg = render(data['camera'], gaussians, data['time'], background, stage="fine")
        rendered_image = render_pkg["render"].unsqueeze(0)

        # Calculate BER on Rendered Image using the (frozen) INN
        # We DO NOT update INN here, but we need gradients to flow through INN to the rendered_image
        w_pred_logits_render = inn_model.extract(rendered_image)[1]  # Get logits only

        # Loss 1: Reconstruction (Visual) -> Match the Watermarked GT (Teacher's output)
        # This helps the Gaussian approximate the watermarked texture
        target_image = wm_image.detach()  # Stop gradients to INN
        loss_recon = l1_loss(rendered_image * valid_mask, target_image * valid_mask)

        # Loss 2: BER Constraint (The critical part for protection)
        # Force the rendered image to be decodable
        loss_ber_gs = compute_ber_loss(w_pred_logits_render, watermark_key)

        # Total Gaussian Loss
        loss_gs = loss_recon + lambda_ber * loss_ber_gs

        loss_gs.backward()
        gaussians.optimizer.step()

        # --- Metrics & Logging ---
        with torch.no_grad():
            # Check accuracy on the rendered image
            acc_w = compute_accuracy(w_pred_logits_render, watermark_key)

        if iteration % 10 == 0:
            progress_bar.set_postfix({
                "INN_L": f"{loss_inn.item():.4f}",
                "Rec_L": f"{loss_recon.item():.4f}",
                "BER_L": f"{loss_ber_gs.item():.4f}",
                "Acc_W": f"{acc_w * 100:.1f}%"
            })

    # Save Final Watermarked Model
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
                        help="Path to clean pretrained model directory (containing point_cloud.ply)")
    parser.add_argument('--workspace', type=str, default='output/watermarked_model', help="Where to save result")
    parser.add_argument('--watermark_iters', type=int, default=5000, help="Number of fine-tuning iterations")
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1])

    # Gaussian params (Default values for initialization)
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