import torch
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
    print(f"Loading INN from {opt.inn_path}")
    inn_model = WatermarkINN().cuda()
    inn_model.load_state_dict(torch.load(os.path.join(opt.inn_path, "best_inn.pth")))
    inn_model.eval()
    # [关键点] 彻底冻结参数，不计算梯度，加速训练
    for param in inn_model.parameters():
        param.requires_grad = False

    watermark_key = torch.load(os.path.join(opt.inn_path, "watermark_key.pth")).cuda()

    # 3. 设置 Gaussian 优化器 (降低学习率)
    gaussians.training_setup()
    for param_group in gaussians.optimizer.param_groups:
        param_group['lr'] *= 0.5

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = 0
    max_iter = opt.watermark_iters

    # [修复进度条] 让进度条对象驱动循环
    progress_bar = tqdm(range(iter_start, max_iter), desc="GS Fine-tuning")

    # 平滑指标记录
    ema_psnr, ema_ssim, ema_acc, ema_loss = 0.0, 0.0, 0.0, 0.0
    lambda_ber = 0.1  # 控制水印权重

    for iteration in progress_bar:
        try:
            data = next(iter(dataloader))
        except StopIteration:
            dataloader_iter = iter(dataloader)
            data = next(dataloader_iter)

        gt_image = data['camera'].original_image.cuda().unsqueeze(0)
        mask = data['mask'].cuda().unsqueeze(0).unsqueeze(0)

        # --- 渲染与 Loss 计算 ---
        gaussians.optimizer.zero_grad(set_to_none=True)

        # 1. 渲染 (Render)
        render_pkg = render(data['camera'], gaussians, data['time'], background, stage="fine")
        rendered_image = render_pkg["render"].unsqueeze(0)

        # 2. 水印提取 (通过冻结的 INN)
        # 梯度流向：BER Loss -> Logits -> Rendered Image -> Gaussian Parameters
        _, w_logits = inn_model.extract(rendered_image)

        # 3. 混合 Loss
        loss_recon = l1_loss(rendered_image * mask, gt_image * mask)
        loss_ber = compute_ber_loss(w_logits, watermark_key)

        loss = loss_recon + lambda_ber * loss_ber
        loss.backward()
        gaussians.optimizer.step()

        # --- 日志记录 ---
        with torch.no_grad():
            acc = compute_accuracy(w_logits, watermark_key)
            # 计算 masked PSNR
            cur_psnr = psnr(rendered_image * mask, gt_image * mask).mean().double().item()
            cur_ssim = ssim(rendered_image * mask, gt_image * mask).mean().item()

            # 指标平滑更新
            ema_psnr = 0.4 * cur_psnr + 0.6 * ema_psnr
            ema_ssim = 0.4 * cur_ssim + 0.6 * ema_ssim
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
    # 为了方便测试脚本，把 INN 权重也拷贝(或重新保存)一份到模型目录
    torch.save(inn_model.state_dict(), os.path.join(opt.workspace, "watermark_inn.pth"))
    torch.save(watermark_key, os.path.join(opt.workspace, "watermark_key.pth"))
    print(f"Fine-tuning complete. Model saved to {opt.workspace}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Dataset path")
    parser.add_argument('--pretrained_model_path', type=str, required=True, help="Clean model path (Step 1 output)")
    parser.add_argument('--inn_path', type=str, required=True, help="Pretrained INN path (Step 2 output)")
    parser.add_argument('--workspace', type=str, default='output/watermarked_gs', help="Final output path")
    parser.add_argument('--watermark_iters', type=int, default=5000)
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1])

    # GS 默认参数 (用于初始化)
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