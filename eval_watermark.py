import torch
import argparse
import os
import math
from tqdm import tqdm

from gaussian_core.provider import EndoDataset
from gaussian_core.utils import seed_everything
from gaussian_core.gaussian_model import GaussianModel
from gaussian_renderer import render
from pytorch_wavelets import DWTForward
import torch.nn.functional as F


def bit_acc(pred, target):
    same = ~torch.logical_xor(pred > 0, target > 0)
    return torch.sum(same) / same.shape[-1]


def eval_simple_wm(opt, dataloader, gaussians):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载带水印的 EndoGS 权重
    print(f"Loading watermarked EndoGS model from {opt.model_path}")
    gaussians.load_ply(os.path.join(opt.model_path, "point_cloud.ply"))
    gaussians.load_model(opt.model_path)

    # 2. 加载我们的 "极简解码器" 密钥
    key_path = os.path.join(opt.workspace, "secret_key.pt")
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"Key file not found at {key_path}. Run training first.")

    key_data = torch.load(key_path, map_location=device)
    projection_matrix = key_data['P']
    secret_msg = key_data['msg']

    dwt_forward = DWTForward(wave='bior4.4', J=2, mode='periodization').to(device)
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    avg_bitacc = 0.
    avg_psnr = 0.
    num_frames = 0

    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader, desc="Extracting Simple Watermark")):
            viewpoint_cam = data['camera']
            time = data['time'].to(device)

            # 渲染图像
            rendering = render(viewpoint_cam, gaussians, time, background, stage="fine")["render"]
            image_wm = rendering.unsqueeze(0).contiguous()
            gt_image = viewpoint_cam.original_image[:3, :, :].unsqueeze(0).to(device)

            # ======== 极简水印提取逻辑 ========
            LL_img, _ = dwt_forward(image_wm)
            LL_pooled = F.adaptive_avg_pool2d(LL_img, (16, 16))
            LL_flat = LL_pooled.view(-1)

            # 预测信息
            pred_msg = torch.matmul(projection_matrix, LL_flat)
            # ==================================

            bit_accu = bit_acc(pred_msg, secret_msg).item()
            mse = F.mse_loss(image_wm, gt_image)
            psnr_val = -10.0 * math.log10(mse)

            avg_bitacc += bit_accu
            avg_psnr += psnr_val
            num_frames += 1

    avg_bitacc /= num_frames
    avg_psnr /= num_frames

    print("\n" + "=" * 50)
    print(f"Evaluation Complete across {num_frames} frames:")
    print(f"Average Bit Accuracy: {avg_bitacc * 100:.2f}%")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Dataset path")
    parser.add_argument('--model_path', type=str, required=True, help="Path to watermarked EndoGS model")
    parser.add_argument('--workspace', type=str, default='workspace_simple_wm',
                        help="Path where secret_key.pt is saved")
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1])
    parser.add_argument('--seed', type=int, default=42)

    opt = parser.parse_args()
    seed_everything(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gaussians = GaussianModel(opt)
    dataloader = EndoDataset(opt, device=device, type='test').dataloader()

    eval_simple_wm(opt, dataloader, gaussians)