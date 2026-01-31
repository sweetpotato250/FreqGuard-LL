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
from utils.loss_utils import l1_loss
from utils.image_utils import psnr


# ==============================================================================
# SECTION 1: 内置 Watermark Core (DWT, INN, Logic)
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
        x_HH = x1 - x2 - x3 + x4
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
        self.split_len1 = in_channels // 2
        self.split_len2 = in_channels - self.split_len1

        self.net = nn.Sequential(
            nn.Conv2d(self.split_len1, mid_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(mid_channels, self.split_len2 * 2, 3, 1, 1)
        )

    def forward(self, x, reverse=False):
        x1, x2 = torch.split(x, [self.split_len1, self.split_len2], dim=1)
        st = self.net(x1)
        s, t = torch.chunk(st, 2, dim=1)
        # 限制 scaling 范围防止数值不稳定
        s = torch.sigmoid(s + 2) + 1e-6

        if not reverse:
            y1 = x1
            y2 = s * x2 + t
            return torch.cat([y1, y2], dim=1)
        else:
            # 严格可逆公式: x2 = (y2 - t) / s
            y1 = x1
            y2 = x2
            x2_restored = (y2 - t) / s
            return torch.cat([y1, x2_restored], dim=1)


class WatermarkINN(nn.Module):
    def __init__(self, in_channels=12, steps=4, wm_len=64, alpha=0.1):
        super(WatermarkINN, self).__init__()
        self.dwt = DWT()
        self.idwt = IDWT()
        self.wm_len = wm_len
        self.alpha = alpha  # 水印注入强度系数

        self.layers = nn.ModuleList([CouplingLayer(in_channels) for _ in range(steps)])

        # 将水印投影到潜空间 (Batch, wm_len) -> (Batch, C)
        self.wm_projector = nn.Sequential(
            nn.Linear(wm_len, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )

        # 从潜空间提取水印 (Batch, C) -> (Batch, wm_len)
        self.wm_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, wm_len)
        )

    def get_wm_feature(self, watermark, dims):
        B, C, H, W = dims
        wm_feat = self.wm_projector(watermark).unsqueeze(-1).unsqueeze(-1)
        return wm_feat.expand(B, C, H, W)

    def inn_forward(self, x):
        for layer in self.layers:
            x = layer(x, reverse=False)
        return x

    def inn_inverse(self, z):
        for layer in reversed(self.layers):
            z = layer(z, reverse=True)
        return z

    def embed(self, image, watermark):
        # 1. Image -> DWT
        coeffs = self.dwt(image)
        # 2. INN Forward -> Latent Z
        z = self.inn_forward(coeffs)
        # 3. Inject Watermark: Z_wm = Z + alpha * W
        wm_feat = self.get_wm_feature(watermark, z.shape)
        z_watermarked = z + self.alpha * wm_feat
        # 4. INN Inverse
        coeffs_watermarked = self.inn_inverse(z_watermarked)
        # 5. IDWT -> Watermarked Image
        return self.idwt(coeffs_watermarked)

    def extract(self, watermarked_image, gt_watermark=None):
        # 1. Image_wm -> DWT
        coeffs_wm = self.dwt(watermarked_image)
        # 2. INN Forward -> Latent Z (contains watermark)
        z_rec = self.inn_forward(coeffs_wm)

        # 3. Extract Watermark
        w_pred_logits = self.wm_extractor(z_rec)

        # 4. Strip Watermark: Z_clean = Z_rec - alpha * W
        # 训练时用 GT 引导剥离，测试时用预测值剥离
        if gt_watermark is not None:
            w_to_subtract = gt_watermark
        else:
            w_to_subtract = (torch.sigmoid(w_pred_logits) > 0.5).float()

        wm_feat_neg = self.get_wm_feature(w_to_subtract, z_rec.shape)
        z_clean = z_rec - self.alpha * wm_feat_neg

        # 5. INN Inverse
        coeffs_clean = self.inn_inverse(z_clean)
        # 6. IDWT -> Restored Image (Lossless-like)
        restored_image = self.idwt(coeffs_clean)

        return restored_image, w_pred_logits


def compute_ber_loss(w_pred_logits, w_gt):
    return F.binary_cross_entropy_with_logits(w_pred_logits, w_gt)


def compute_accuracy(w_pred_logits, w_gt):
    w_pred_bits = (torch.sigmoid(w_pred_logits) > 0.5).float()
    correct_bits = (w_pred_bits == w_gt).float()
    return correct_bits.mean().item()


# ==============================================================================
# SECTION 2: 训练辅助函数
# ==============================================================================

def warmup_inn(inn_model, dataloader, watermark_key, epochs=20):
    """
    预热 INN：让它先学会 '复制' 和 '简单的嵌入/提取'
    """
    print(f"\n[Phase 1] Warming up INN for {epochs} epochs...")
    optimizer = optim.Adam(inn_model.parameters(), lr=1e-3)
    inn_model.train()

    for epoch in range(epochs):
        epoch_acc = 0
        count = 0
        limit_batches = 50  # 限制每轮步数，节省时间

        pbar = tqdm(enumerate(dataloader), total=limit_batches, desc=f"Warmup {epoch + 1}/{epochs}", leave=False)

        for i, data in pbar:
            if i >= limit_batches: break

            gt_image = data['camera'].original_image.cuda().unsqueeze(0)

            # Forward
            wm_image = inn_model.embed(gt_image, watermark_key)
            # 使用 GT key 来训练复原能力
            restored_image, w_logits = inn_model.extract(wm_image, gt_watermark=watermark_key)

            # Loss:
            # 1. 嵌入图要像原图
            # 2. 复原图要严格等于原图 (Critical for Rec_PSNR)
            # 3. 水印要对
            loss_visual = l1_loss(wm_image, gt_image)
            loss_restore = l1_loss(restored_image, gt_image)
            loss_bits = compute_ber_loss(w_logits, watermark_key)

            loss = loss_visual + loss_bits + loss_restore

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = compute_accuracy(w_logits, watermark_key)
            epoch_acc += acc
            count += 1
            pbar.set_postfix({"Acc": f"{acc:.2f}"})

    print("[Phase 1] Warmup complete.\n")
    return inn_model


# ==============================================================================
# SECTION 3: 主训练逻辑
# ==============================================================================

def train_watermark(opt, dataloader, gaussians):
    # 1. 加载 Pretrained Model
    print(f"Loading Gaussians from {opt.pretrained_model_path}")
    gaussians.load_ply(os.path.join(opt.pretrained_model_path, "point_cloud.ply"))
    gaussians.load_model(opt.pretrained_model_path)

    # 2. 初始化 INN
    inn_model = WatermarkINN(wm_len=opt.wm_len, alpha=0.1).cuda()

    # 3. 生成 Key
    watermark_key = torch.randint(0, 2, (1, opt.wm_len)).float().cuda()
    print(f"Generated Key: {watermark_key[0, :10].cpu().numpy()}...")

    # 4. 热身 INN
    inn_model = warmup_inn(inn_model, dataloader, watermark_key, epochs=20)

    # 5. 配置联合训练优化器
    print("Configuring Optimizers for Joint Training...")
    # INN 优化器 (使用较小的 LR 微调)
    inn_optimizer = optim.Adam(inn_model.parameters(), lr=1e-4)
    inn_model.train()

    # GS 优化器设置
    gaussians.training_setup()
    hot_params = ['f_dc', 'f_rest']  # 只微调颜色相关，冻结几何
    for param_group in gaussians.optimizer.param_groups:
        name = param_group['name']
        if name in hot_params:
            param_group['lr'] *= 0.5
            for param in param_group['params']: param.requires_grad = True
            print(f"  -> [HOT] Active: {name}")
        else:
            param_group['lr'] = 0.0
            for param in param_group['params']: param.requires_grad = False

    # 6. 主循环
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = 0
    max_iter = opt.watermark_iters
    progress_bar = tqdm(range(iter_start, max_iter), desc="[Phase 2] Joint Training")

    ema_psnr, ema_acc, ema_rec = 0.0, 0.0, 0.0
    warned_missing_mask = False

    # Loss 权重
    lambda_ber = 0.1
    lambda_restore = 2.0  # 高权重以保证 Rec_PSNR

    for iteration in progress_bar:
        try:
            data = next(iter(dataloader))
        except StopIteration:
            dataloader_iter = iter(dataloader)
            data = next(dataloader_iter)

        gt_image = data['camera'].original_image.cuda().unsqueeze(0)
        B, C, H, W = gt_image.shape

        # --- Mask 处理 (保证尺寸一致) ---
        tool_mask = data['mask'].cuda().unsqueeze(0).unsqueeze(0).float()
        if tool_mask.shape[2] != H or tool_mask.shape[3] != W:
            tool_mask = F.interpolate(tool_mask, size=(H, W), mode='nearest')

        if 'lesion_mask' in data:
            lesion_mask = data['lesion_mask'].cuda().unsqueeze(0).unsqueeze(0).float()
            if lesion_mask.shape[2] != H or lesion_mask.shape[3] != W:
                lesion_mask = F.interpolate(lesion_mask, size=(H, W), mode='nearest')
        else:
            if not warned_missing_mask and iteration == 0:
                print(f"[WARN] No lesion_mask found. Using Ones.")
                warned_missing_mask = True
            lesion_mask = torch.ones_like(tool_mask)

        # RONI Mask (Region of Non-Interest)
        roni_mask = tool_mask * lesion_mask
        erosion_kernel = 21
        inverted_roni = 1.0 - roni_mask
        dilated_inverted = F.max_pool2d(inverted_roni, kernel_size=erosion_kernel, stride=1,
                                        padding=erosion_kernel // 2)
        final_embed_mask = 1.0 - dilated_inverted

        # --- 训练步 ---
        gaussians.optimizer.zero_grad(set_to_none=True)
        inn_optimizer.zero_grad()

        # 1. Render GS
        render_pkg = render(data['camera'], gaussians, data['time'], background, stage="fine")
        rendered_image = render_pkg["render"].unsqueeze(0)

        # 2. 生成目标图像 (Ground Truth + Watermark)
        # 用 INN 生成一个 "完美的水印图" 作为 GS 的学习目标
        wm_full = inn_model.embed(gt_image, watermark_key)
        target_image = wm_full * final_embed_mask + gt_image * (1.0 - final_embed_mask)

        # 3. 提取与复原 (从 GS 渲染图出发)
        # 传入 gt_watermark=watermark_key 指导训练时的剥离
        restored_image, w_logits = inn_model.extract(rendered_image, gt_watermark=watermark_key)

        # --- Loss ---
        # L1: GS 应该渲染出带水印的图
        loss_gs = l1_loss(rendered_image * tool_mask, target_image * tool_mask)

        # L2: 水印应该能被提取
        loss_ber = compute_ber_loss(w_logits, watermark_key)

        # L3: 剥离水印后应该等于原图 (Rec Loss)
        # 注意：这里对比的是 restored_image 和 gt_image
        loss_rec = l1_loss(restored_image * tool_mask, gt_image * tool_mask)

        loss = loss_gs + lambda_ber * loss_ber + lambda_restore * loss_rec

        loss.backward()
        gaussians.optimizer.step()
        inn_optimizer.step()

        # --- Logs ---
        with torch.no_grad():
            acc = compute_accuracy(w_logits, watermark_key)
            cur_psnr = psnr(rendered_image * tool_mask, gt_image * tool_mask).mean().double().item()
            cur_rec = psnr(restored_image * tool_mask, gt_image * tool_mask).mean().double().item()

            ema_psnr = 0.4 * cur_psnr + 0.6 * ema_psnr
            ema_acc = 0.4 * acc + 0.6 * ema_acc
            ema_rec = 0.4 * cur_rec + 0.6 * ema_rec

        if iteration % 10 == 0:
            progress_bar.set_postfix({
                "PSNR": f"{ema_psnr:.2f}",
                "Rec": f"{ema_rec:.2f}",  # 这里的 Rec 应该很高
                "Acc": f"{ema_acc:.2f}"
            })

    # 保存模型
    print(f"\nSaving to {opt.workspace}...")
    os.makedirs(opt.workspace, exist_ok=True)
    gaussians.save(opt.workspace, max_iter, "fine")

    ckpt_dir = os.path.join(opt.workspace, "point_cloud", f"iteration_{max_iter}")
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save(inn_model.state_dict(), os.path.join(ckpt_dir, "watermark_inn.pth"))
    torch.save(watermark_key, os.path.join(ckpt_dir, "watermark_key.pth"))
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Dataset path")
    parser.add_argument('--pretrained_model_path', type=str, required=True)
    parser.add_argument('--workspace', type=str, default='output/watermarked_gs')
    parser.add_argument('--watermark_iters', type=int, default=5000)
    parser.add_argument('--wm_len', type=int, default=64)
    # [修复点] 显式添加 data_range 参数
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1], help="Data range to use")

    # GS Params
    parser.add_argument('--sh_degree', type=int, default=3)
    parser.add_argument('--position_lr_init', type=float, default=0.00016)
    parser.add_argument('--feature_lr', type=float, default=0.0025)
    parser.add_argument('--opacity_lr', type=float, default=0.05)
    parser.add_argument('--scaling_lr', type=float, default=0.005)
    parser.add_argument('--rotation_lr', type=float, default=0.001)

    opt, _ = parser.parse_known_args()  # parse_known_args 允许忽略未定义的 GS 参数

    # 补全 GaussianModel 需要的参数 (避免报错)
    opt.percent_dense = 0.01
    opt.position_lr_final = 0.0000016
    opt.position_lr_delay_mult = 0.01
    opt.position_lr_max_steps = 1000000
    opt.grid_lr_init = 0.00015
    opt.grid_lr_final = 0.000015
    opt.deformation_lr_init = 0.000015
    opt.deformation_lr_final = 0.0000015
    opt.deformation_lr_delay_mult = 0.01
    opt.deformation_lr_max_steps = 1000000

    seed_everything(0)

    gaussians = GaussianModel(opt)
    device = torch.device('cuda')
    dataset = EndoDataset(opt, device=device, type='train')
    train_watermark(opt, dataset.dataloader(), gaussians)