import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from tqdm import tqdm
from gaussian_core.provider import EndoDataset
from gaussian_core.utils import seed_everything
from gaussian_core.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
# 导入核心模块
from watermark_core import WatermarkINN, compute_ber_loss, compute_accuracy


def train_watermark_frozen(opt, dataloader, gaussians):
    # 1. 加载第一步训练好的 Clean Gaussian Model
    print(f"Loading Gaussians from {opt.pretrained_model_path}")
    gaussians.load_ply(os.path.join(opt.pretrained_model_path, "point_cloud.ply"))
    gaussians.load_model(opt.pretrained_model_path)

    # 2. 加载预训练好的 INN 并冻结
    print(f"Loading INN from {opt.inn_path} with wm_len={opt.wm_len}")
    inn_model = WatermarkINN(wm_len=opt.wm_len).cuda()

    inn_ckpt = os.path.join(opt.inn_path, "best_inn.pth")
    if not os.path.exists(inn_ckpt):
        raise FileNotFoundError(f"INN Checkpoint not found: {inn_ckpt}")

    inn_model.load_state_dict(torch.load(inn_ckpt))
    inn_model.eval()
    # [关键点] 彻底冻结参数
    for param in inn_model.parameters():
        param.requires_grad = False

    key_ckpt = os.path.join(opt.inn_path, "watermark_key.pth")
    watermark_key = torch.load(key_ckpt).cuda()

    # 检查 Key 长度是否匹配
    if watermark_key.shape[1] != opt.wm_len:
        raise ValueError(
            f"Key length mismatch! Loaded key has {watermark_key.shape[1]} bits, but --wm_len is {opt.wm_len}")

    # 3. 设置 Gaussian 优化器 (降低学习率)
    gaussians.training_setup()
    for param_group in gaussians.optimizer.param_groups:
        param_group['lr'] *= 0.5

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = 0
    max_iter = opt.watermark_iters

    progress_bar = tqdm(range(iter_start, max_iter), desc="GS Fine-tuning")

    ema_psnr, ema_ssim, ema_acc, ema_loss = 0.0, 0.0, 0.0, 0.0
    lambda_ber = 0.1

    for iteration in progress_bar:
        try:
            data = next(iter(dataloader))
        except StopIteration:
            dataloader_iter = iter(dataloader)
            data = next(dataloader_iter)

        gt_image = data['camera'].original_image.cuda().unsqueeze(0)
        mask = data['mask'].cuda().unsqueeze(0).unsqueeze(0)

        # =========================================================
        # [核心修改] Mask 腐蚀策略 (Erosion Strategy)
        # =========================================================
        # 原理：使用 MaxPool 对 Mask 取反来实现腐蚀，把有效区域向内缩
        # kernel_size=21 表示向内缩大约 10 个像素，避开手术钳边缘的伪影
        erosion_kernel = 21

        # 1. 取反 (因为 MaxPool 是膨胀，我们需要腐蚀，所以先反过来)
        inverted_mask = 1.0 - mask
        # 2. 膨胀 (让非组织区域变大)
        dilated_inverted_mask = F.max_pool2d(inverted_mask, kernel_size=erosion_kernel, stride=1,
                                             padding=erosion_kernel // 2)
        # 3. 再取反回来 (得到缩小了的组织 Mask)
        eroded_mask = 1.0 - dilated_inverted_mask

        # 使用腐蚀后的 Mask 作为水印嵌入的“安全区”
        valid_mask = eroded_mask
        # =========================================================

        # --- 渲染与 Loss 计算 ---
        gaussians.optimizer.zero_grad(set_to_none=True)

        # 1. 渲染
        render_pkg = render(data['camera'], gaussians, data['time'], background, stage="fine")
        rendered_image = render_pkg["render"].unsqueeze(0)

        # 2. 水印提取 (通过冻结的 INN)
        _, w_logits = inn_model.extract(rendered_image)

        # 3. 混合 Loss
        # [关键] 只有在“安全区”(valid_mask) 内，我们才要求 Gaussian 拟合水印
        loss_recon = l1_loss(rendered_image * valid_mask, gt_image * valid_mask)
        loss_ber = compute_ber_loss(w_logits, watermark_key)

        loss = loss_recon + lambda_ber * loss_ber
        loss.backward()
        gaussians.optimizer.step()

        # --- 日志记录 ---
        with torch.no_grad():
            acc = compute_accuracy(w_logits, watermark_key)
            # 计算 PSNR 时也只看 Mask 区域 (这里用原 Mask 或者腐蚀后的 Mask 都可以，用原 Mask 更能反映整体观感)
            cur_psnr = psnr(rendered_image * mask, gt_image * mask).mean().double().item()

            ema_psnr = 0.4 * cur_psnr + 0.6 * ema_psnr
            ema_acc = 0.4 * acc + 0.6 * ema_acc
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss

        if iteration % 10 == 0:
            progress_bar.set_postfix({
                "Loss": f"{ema_loss:.4f}",
                "PSNR": f"{ema_psnr:.2f}",
                "Acc": f"{ema_acc:.2f}"
            })

    # 保存最终结果
    os.makedirs(opt.workspace, exist_ok=True)
    gaussians.save(opt.workspace, max_iter, "fine")

    # 拷贝 INN 和 Key 到结果目录
    torch.save(inn_model.state_dict(), os.path.join(opt.workspace, "watermark_inn.pth"))
    torch.save(watermark_key, os.path.join(opt.workspace, "watermark_key.pth"))
    print(f"Fine-tuning complete. Model saved to {opt.workspace}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Dataset path")
    parser.add_argument('--pretrained_model_path', type=str, required=True, help="Clean model path")
    parser.add_argument('--inn_path', type=str, required=True, help="Pretrained INN path")
    parser.add_argument('--workspace', type=str, default='output/watermarked_gs', help="Final output path")
    parser.add_argument('--wm_len', type=int, default=64, help="Watermark length (default: 64)")
    parser.add_argument('--watermark_iters', type=int, default=5000)
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1])

    # GS 默认参数
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

    train_watermark_frozen(opt, dataloader, gaussians)