import torch
import os
import argparse
from tqdm import tqdm
import torchvision
import numpy as np
import imageio  # [新增] 用于生成视频

from gaussian_core.provider import EndoDataset
from gaussian_core.utils import seed_everything
from gaussian_core.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.loss_utils import ssim
from utils.image_utils import psnr

# 导入核心模块
from watermark_core import WatermarkINN, compute_accuracy


def tensor2numpy(tensor):
    """
    辅助函数：将 Tensor (1, 3, H, W) 转换为 Numpy (H, W, 3) 且范围在 [0, 255] 的 uint8
    """
    img = tensor.squeeze(0).cpu().clamp(0, 1).numpy()
    img = np.transpose(img, (1, 2, 0))  # C,H,W -> H,W,C
    return (img * 255).astype(np.uint8)


def test_watermark(opt):
    model_path = opt.model_path
    if not model_path.endswith("/"): model_path += "/"

    if opt.output_path is not None:
        output_dir = opt.output_path
    else:
        output_dir = model_path
    os.makedirs(output_dir, exist_ok=True)

    # 路径设置
    render_dir = os.path.join(output_dir, "renders")
    gt_dir = os.path.join(output_dir, "gt")

    if opt.save_images:
        os.makedirs(render_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

    print(f"[INFO] Loading Watermarked Gaussians from: {model_path}")
    print(f"[INFO] Watermark Length: {opt.wm_len}")

    # 1. 加载 Gaussian
    gaussians = GaussianModel(opt)
    gaussians.load_ply(os.path.join(model_path, "point_cloud.ply"))
    gaussians.load_model(model_path)

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 2. 加载 INN
    print(f"[INFO] Loading INN Decoder and Key...")
    inn_model = WatermarkINN(wm_len=opt.wm_len).cuda()

    inn_ckpt_path = os.path.join(model_path, "watermark_inn.pth")
    key_ckpt_path = os.path.join(model_path, "watermark_key.pth")

    if not os.path.exists(inn_ckpt_path):
        raise FileNotFoundError(f"Missing watermark_inn.pth in {model_path}")

    inn_model.load_state_dict(torch.load(inn_ckpt_path))
    inn_model.eval()

    # 加载水印 Key
    watermark_key = torch.load(key_ckpt_path).cuda()
    key_bits = watermark_key[0].cpu().detach().numpy().astype(int)
    key_str = "".join(str(b) for b in key_bits)
    print(f"[INFO] Watermark Message: {key_str}")

    if watermark_key.shape[1] != opt.wm_len:
        print(
            f"[WARNING] Loaded key length ({watermark_key.shape[1]}) != --wm_len ({opt.wm_len}). Using loaded key length.")

    # 3. 准备数据
    device = torch.device('cuda')
    dataset = EndoDataset(opt, device=device, type='test')
    dataloader = dataset.dataloader()

    output_file = os.path.join(output_dir, "test_metrics_report.txt")

    total_psnr, total_ssim, total_acc = 0.0, 0.0, 0.0
    count = 0

    # [新增] 视频写入器初始化
    video_writer = None
    if opt.save_video:
        video_path = os.path.join(output_dir, "render_video.mp4")
        print(f"[INFO] Video recording enabled. Saving to: {video_path}")
        # macro_block_size=1 避免有些播放器因为分辨率不能被16整除而报错
        video_writer = imageio.get_writer(video_path, fps=opt.fps, macro_block_size=1)

    with open(output_file, "w") as f:
        f.write(f"Watermark_Message (Len={len(key_str)}): {key_str}\n")
        f.write("=" * 80 + "\n")
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

                # Extract
                _, w_logits = inn_model.extract(rendered_image)

                # Metrics
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

                # 保存单帧图片
                if opt.save_images:
                    torchvision.utils.save_image(rendered_image, os.path.join(render_dir, f"{image_id}.png"))
                    torchvision.utils.save_image(gt_image, os.path.join(gt_dir, f"{image_id}.png"))

                # [新增] 写入视频帧
                if video_writer is not None:
                    # 转换 Tensor 为 Numpy 图片
                    frame_uint8 = tensor2numpy(rendered_image)
                    video_writer.append_data(frame_uint8)

                f.write(f"{image_id:<15} | {cur_psnr:<10.4f} | {cur_ssim:<10.4f} | {cur_acc:<10.4f}\n")

        f.write("-" * 55 + "\n")
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        avg_acc = total_acc / count
        f.write(f"{'AVERAGE':<15} | {avg_psnr:<10.4f} | {avg_ssim:<10.4f} | {avg_acc:<10.4f}\n")

    # [新增] 关闭视频流
    if video_writer is not None:
        video_writer.close()
        print(f"[DONE] Video saved to: {video_path}")

    print(f"\n[DONE] Avg PSNR: {avg_psnr:.4f} | Avg SSIM: {avg_ssim:.4f} | Avg Acc: {avg_acc * 100:.2f}%")
    print(f"Report saved to: {output_file}")
    if opt.save_images:
        print(f"Images saved to: {render_dir} and {gt_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Dataset path")
    parser.add_argument('--model_path', type=str, required=True, help="Watermarked model path")
    parser.add_argument('--output_path', type=str, default=None, help="Output path")

    # [修改] 增加视频相关参数
    parser.add_argument('--save_images', action='store_true', help="Save individual frames as png")
    parser.add_argument('--save_video', action='store_true', help="Save rendered video (mp4)")
    parser.add_argument('--fps', type=int, default=30, help="Video FPS")

    parser.add_argument('--wm_len', type=int, default=64, help="Watermark length (default: 64)")
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1])

    # GS Init Params
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