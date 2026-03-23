import torch
import argparse
import os
import math
import torch.nn.functional as F
from tqdm import tqdm

# EndoGS 依赖
from gaussian_core.provider import EndoDataset
from gaussian_core.utils import seed_everything
from gaussian_core.gaussian_model import GaussianModel
from gaussian_renderer import render

# 3D-GSW 依赖
from decoder.init_decoder import DecoderAttributes
from pytorch_wavelets import DWTForward
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import lpips


def bit_acc(decoded, keys):
    diff = (~torch.logical_xor(decoded > 0, keys > 0))
    bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]
    return bit_accs


def finetune_watermark(opt, dataloader, gaussians):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建输出目录与日志文件
    os.makedirs(opt.workspace, exist_ok=True)
    log_path = os.path.join(opt.workspace, "watermark_train_log.txt")
    with open(log_path, "w") as log_file:
        log_file.write("Iteration,PSNR,Bit_Accuracy,BER,WM_Loss\n")
    print(f"Logging training metrics to: {log_path}")

    # 1. 加载 EndoGS 预训练权重
    print(f"Loading pre-trained EndoGS model from {opt.model_path}")
    gaussians.load_ply(os.path.join(opt.model_path, "point_cloud.ply"))
    gaussians.load_model(opt.model_path)
    gaussians.training_setup()  # 初始化优化器

    # 2. 初始化 3D-GSW 水印 Decoder
    dec_attrs = DecoderAttributes(opt.decoder_att, opt.seed)
    print(f'Loading decoder from {dec_attrs.decoder_path}...')
    msg_decoder = dec_attrs.dec.to(device)
    msg_decoder.eval()
    loss_type = dec_attrs.loss_dict
    gt_msg = dec_attrs.msg.to(device)  # 目标水印信息

    bg_color = [0, 0, 0]  # 内窥镜一般背景为黑
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    alpha = 1.0  # 控制梯度掩码的超参数

    progress_bar = tqdm(range(opt.finetune_iters), desc="Watermark Finetuning")

    # 冻结 Deformation 网络，只微调基础高斯属性以嵌入水印
    for param in gaussians._deformation.parameters():
        param.requires_grad = False

    iteration = 0
    dwt_forward = DWTForward(wave='bior4.4', J=2, mode='periodization').to(device)

    # 转换为无限循环的迭代器
    data_iterator = iter(dataloader)

    while iteration < opt.finetune_iters:
        try:
            data = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            data = next(data_iterator)

        iteration += 1
        gaussians.update_learning_rate(iteration)

        viewpoint_cam = data['camera']
        time = data['time'].to(device)

        # EndoGS 渲染 (带时间参数)
        render_pkg = render(viewpoint_cam, gaussians, time, background, stage="fine")
        image = render_pkg["render"].unsqueeze(0).contiguous()

        gt_image = viewpoint_cam.original_image.to(device).unsqueeze(0).contiguous()

        # DWT 变换
        LL_img, yh = dwt_forward(image)
        LL_img_gt, yh_gt = dwt_forward(gt_image)

        # 提取水印并计算损失
        decoded = msg_decoder(LL_img)
        loss_wm = loss_type['loss_w'](decoded, gt_msg)

        # 图像质量损失
        loss_im_mse = F.mse_loss(gt_image, image)
        psnr = -10.0 * math.log10(loss_im_mse)
        Ll1 = l1_loss(image, gt_image)
        loss_lpips = lpips(image, gt_image, net_type='vgg')

        # DWT 子带损失
        loss_lhhlhh_mse = torch.mean(torch.abs(yh[1][:, :, 0, :, :] - yh_gt[1][:, :, 0, :, :])) + \
                          torch.mean(torch.abs(yh[0][:, :, 0, :, :] - yh_gt[0][:, :, 0, :, :])) + \
                          torch.mean(torch.abs(yh[1][:, :, 1, :, :] - yh_gt[1][:, :, 1, :, :])) + \
                          torch.mean(torch.abs(yh[0][:, :, 1, :, :] - yh_gt[0][:, :, 1, :, :])) + \
                          torch.mean(torch.abs(yh[1][:, :, 2, :, :] - yh_gt[1][:, :, 2, :, :])) + \
                          torch.mean(torch.abs(yh[0][:, :, 2, :, :] - yh_gt[0][:, :, 2, :, :]))

        # 总损失
        loss = opt.lambda_lpips * loss_lpips + \
               opt.lambda_i * Ll1 + \
               opt.lambda_subband * loss_lhhlhh_mse + \
               opt.lambda_wm * loss_wm

        loss.backward()

        # 3D-GSW 的梯度 Masking (防止破坏重要的几何结构)
        with torch.no_grad():
            w1 = (torch.abs(gaussians._features_dc)) ** alpha
            w2 = (torch.abs(gaussians._features_rest)) ** alpha
            w3 = (torch.abs(gaussians._opacity)) ** alpha
            w4 = (torch.abs(gaussians._rotation)) ** alpha
            w5 = (torch.abs(gaussians.get_scaling)) ** alpha

            msk1 = (1. / torch.exp(w1)) / torch.sum(1. / torch.exp(w1), dim=0)
            msk2 = (1. / torch.exp(w2)) / torch.sum(1. / torch.exp(w2), dim=0)
            msk3 = (1. / torch.exp(w3)) / torch.sum(1. / torch.exp(w3), dim=0)
            msk4 = (1. / torch.exp(w4)) / torch.sum(1. / torch.exp(w4), dim=0)
            msk5 = (1. / torch.exp(w5)) / torch.sum(1. / torch.exp(w5), dim=0)

            if gaussians._features_dc.grad is not None:
                gaussians._features_dc.grad *= msk1
            if gaussians._features_rest.grad is not None:
                gaussians._features_rest.grad *= msk2
            if gaussians._opacity.grad is not None:
                gaussians._opacity.grad *= msk3
            if gaussians._rotation.grad is not None:
                gaussians._rotation.grad *= msk4
            if gaussians._scaling.grad is not None:
                gaussians._scaling.grad *= msk5

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        # 日志记录与打印
        if iteration % 10 == 0:
            bit_accuracy = bit_acc(decoded, gt_msg).item()
            ber = 1.0 - bit_accuracy  # 计算 BER

            # 更新进度条
            progress_bar.set_postfix({"BitAcc": f"{bit_accuracy:.4f}", "BER": f"{ber:.4f}", "PSNR": f"{psnr:.2f}"})
            progress_bar.update(10)

            # 写入 txt 文件
            with open(log_path, "a") as log_file:
                log_file.write(f"{iteration},{psnr:.4f},{bit_accuracy:.4f},{ber:.4f},{loss_wm.item():.4f}\n")

    # 训练结束，保存带有水印的高斯模型
    save_path = os.path.join(opt.workspace, "watermarked_model")
    os.makedirs(save_path, exist_ok=True)
    gaussians.save(save_path, iteration, "fine")
    print(f"\nWatermarked model saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 参数同上...
    parser.add_argument('path', type=str, help="Dataset path")
    parser.add_argument('--model_path', type=str, required=True, help="Path to pre-trained EndoGS")
    parser.add_argument('--workspace', type=str, default='workspace_wm')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1])
    parser.add_argument('--finetune_iters', type=int, default=5000)
    parser.add_argument('--percent_dense', type=float, default=0.01)
    parser.add_argument('--position_lr_init', type=float, default=0.00016)
    parser.add_argument('--position_lr_final', type=float, default=0.0000016)
    parser.add_argument('--position_lr_delay_mult', type=float, default=0.01)
    parser.add_argument('--position_lr_max_steps', type=int, default=30000)
    parser.add_argument('--feature_lr', type=float, default=0.0025)
    parser.add_argument('--opacity_lr', type=float, default=0.05)
    parser.add_argument('--scaling_lr', type=float, default=0.005)
    parser.add_argument('--rotation_lr', type=float, default=0.001)
    parser.add_argument('--grid_lr_init', type=float, default=0.0)
    parser.add_argument('--deformation_lr_init', type=float, default=0.0)
    parser.add_argument("--decoder_att", default="decoder/cfg_32_bce.json", type=str)
    parser.add_argument('--lambda_lpips', type=float, default=0.2)
    parser.add_argument('--lambda_i', type=float, default=0.8)
    parser.add_argument('--lambda_subband', type=float, default=1.0)
    parser.add_argument('--lambda_wm', type=float, default=1.0)

    opt = parser.parse_args()
    seed_everything(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gaussians = GaussianModel(opt)
    dataloader = EndoDataset(opt, device=device, type='train').dataloader()

    finetune_watermark(opt, dataloader, gaussians)