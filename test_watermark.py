import torch
import os
import argparse
from tqdm import tqdm
import torchvision  # 用于保存图片

from gaussian_core.provider import EndoDataset
from gaussian_core.utils import seed_everything
from gaussian_core.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.loss_utils import ssim
from utils.image_utils import psnr

# [重点] 直接从核心文件导入，无需重复定义
from watermark_core import WatermarkINN, compute_accuracy


def test_watermark(opt):
    # 路径处理
    model_path = opt.model_path
    if not model_path.endswith("/"): model_path += "/"

    # 确定输出路径
    if opt.output_path is not None:
        output_dir = opt.output_path
    else:
        output_dir = model_path
    os.makedirs(output_dir, exist_ok=True)

    # 图片保存路径
    render_dir = os.path.join(output_dir, "renders")
    if opt.save_images:
        os.makedirs(render_dir, exist_ok=True)

    print(f"[INFO] Loading Watermarked Gaussians from: {model_path}")
    print(f"[INFO] Results will be saved to: {output_dir}")

    # 1. 加载 Gaussian Model
    gaussians = GaussianModel(opt)
    gaussians.load_ply(os.path.join(model_path, "point_cloud.ply"))
    gaussians.load_model(model_path)

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 2. 加载 INN 解码器 & Key
    # (假设训练脚本最后把 INN 权重也保存到了 workspace 下)
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
    print(f"[INFO] Loaded Key: {watermark_key[0, :8].cpu().numpy().astype(int)}...")

    # 3. 准备测试数据
    device = torch.device('cuda')
    dataset = EndoDataset(opt, device=device, type='test')
    dataloader = dataset.dataloader()

    # 4. 测试循环
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

                # Render
                render_pkg = render(data['camera'], gaussians, data['time'], background, stage="fine")
                rendered_image = render_pkg["render"].unsqueeze(0)

                # Extract Watermark
                _, w_logits = inn_model.extract(rendered_image)

                # Compute Metrics (Masked)
                masked_render = rendered_image * mask
                masked_gt = gt_image * mask

                cur_psnr = psnr(masked_render, masked_gt).mean().double().item()
                cur_ssim = ssim(masked_render, masked_gt).mean().item()
                cur_acc = compute_accuracy(w_logits, watermark_key)

                total_psnr += cur_psnr
                total_ssim += cur_ssim
                total_acc += cur_acc
                count += 1

                # Image ID
                image_id = f"img_{i:04d}"
                if hasattr(data['camera'], 'image_name'):
                    image_id = data['camera'].image_name

                # Save Image (Optional)
                if opt.save_images:
                    torchvision.utils.save_image(rendered_image, os.path.join(render_dir, f"{image_id}.png"))

                f.write(f"{image_id:<15} | {cur_psnr:<10.4f} | {cur_ssim:<10.4f} | {cur_acc:<10.4f}\n")

        # Summary
        f.write("-" * 55 + "\n")
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        avg_acc = total_acc / count
        f.write(f"{'AVERAGE':<15} | {avg_psnr:<10.4f} | {avg_ssim:<10.4f} | {avg_acc:<10.4f}\n")

    print(f"\n[DONE] Avg PSNR: {avg_psnr:.4f} | Avg SSIM: {avg_ssim:.4f} | Avg Acc: {avg_acc * 100:.2f}%")
    print(f"Report saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Dataset path")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the watermarked model folder")
    parser.add_argument('--output_path', type=str, default=None, help="Optional: Path to save results")
    parser.add_argument('--save_images', action='store_true', help="Save rendered images")
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1])

    # 必要的初始化参数 (无需修改)
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