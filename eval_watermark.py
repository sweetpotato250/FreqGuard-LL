import torch
import argparse
import os
import torchvision.transforms as transforms
from tqdm import tqdm

from gaussian_core.provider import EndoDataset
from gaussian_core.utils import seed_everything
from gaussian_core.gaussian_model import GaussianModel
from gaussian_renderer import render

from decoder.init_decoder import DecoderAttributes
from pytorch_wavelets import DWTForward
from utils.image_utils import psnr


def bit_acc(gt, pred):
    same = ~torch.logical_xor(gt > 0, pred > 0)
    bit_accs = torch.sum(same) / same.shape[-1]
    return bit_accs


def test_watermark(opt, dataloader, gaussians):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # 1. 加载带水印的 EndoGS 权重
    print(f"Loading watermarked EndoGS model from {opt.model_path}")
    gaussians.load_ply(os.path.join(opt.model_path, "point_cloud.ply"))
    gaussians.load_model(opt.model_path)

    # 创建日志文件
    log_dir = os.path.dirname(opt.model_path)
    log_path = os.path.join(log_dir, "watermark_eval_log.txt")
    with open(log_path, "w") as log_file:
        log_file.write("Frame_ID,PSNR,Bit_Accuracy,BER\n")
    print(f"Logging evaluation metrics to: {log_path}")

    # 2. 初始化 3D-GSW 水印 Decoder
    dec_attrs = DecoderAttributes(cfg_path=opt.decoder_att, seed=opt.seed)
    msg_decoder = dec_attrs.dec.to(device)
    msg_decoder.eval()

    # 提取真实 Secret Message
    with open(dec_attrs.message_path, 'r') as f:
        key = f.read().strip()
        keyarr = [int(_) for _ in key]
        keytensor = torch.tensor(keyarr, dtype=torch.float32).to(device)[None, :]

    dwt_forward = DWTForward(wave='bior4.4', J=2, mode='periodization').to(device)

    avg_test_bitacc = 0.
    avg_psnr = 0.
    num_frames = 0

    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader, desc="Extracting Watermarks")):
            viewpoint_cam = data['camera']
            time = data['time'].to(device)

            # 渲染出图
            rendering = render(viewpoint_cam, gaussians, time, background, stage="fine")["render"]
            image_wm = rendering.unsqueeze(0).contiguous()
            gt_image = viewpoint_cam.original_image[:3, :, :].unsqueeze(0).to(device)

            # 使用 DWT 获取 LL 子带并解码
            LL_img, _ = dwt_forward(image_wm)
            pred_msg = msg_decoder(LL_img)

            # 计算指标
            bit_accu = bit_acc(keytensor, pred_msg).item()
            ber = 1.0 - bit_accu  # 计算误码率 (BER)
            psnr_val = psnr(image_wm, gt_image).item()

            avg_test_bitacc += bit_accu
            avg_psnr += psnr_val
            num_frames += 1

            # 写入单帧测试结果
            with open(log_path, "a") as log_file:
                log_file.write(f"{idx},{psnr_val:.4f},{bit_accu:.4f},{ber:.4f}\n")

    avg_test_bitacc /= num_frames
    avg_ber = 1.0 - avg_test_bitacc
    avg_psnr /= num_frames

    # 在日志文件末尾写入汇总信息
    with open(log_path, "a") as log_file:
        log_file.write("\n========================================\n")
        log_file.write(f"Summary across {num_frames} test frames:\n")
        log_file.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
        log_file.write(f"Average Bit Accuracy: {avg_test_bitacc * 100:.2f}%\n")
        log_file.write(f"Average BER: {avg_ber * 100:.2f}%\n")
        log_file.write("========================================\n")

    print("\n" + "=" * 50)
    print(f"Evaluation Complete! Results saved to {log_path}")
    print(f"Average Bit Accuracy: {avg_test_bitacc * 100:.2f}%")
    print(f"Average BER: {avg_ber * 100:.2f}%")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Dataset path")
    parser.add_argument('--model_path', type=str, required=True, help="Path to watermarked EndoGS model")
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--decoder_att", default="decoder/cfg_32_bce.json", type=str,
                        help="Path to decoder config json")

    seed_everything(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gaussians = GaussianModel(opt)
    # 使用测试集来评估水印的鲁棒性
    dataloader = EndoDataset(opt, device=device, type='test').dataloader()

    test_watermark(opt, dataloader, gaussians)