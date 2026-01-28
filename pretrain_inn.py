import torch
import torch.optim as optim
import os
import argparse
from tqdm import tqdm
from gaussian_core.provider import EndoDataset
from gaussian_core.utils import seed_everything
from utils.loss_utils import l1_loss
# 导入我们拆分出来的核心模块
from watermark_core import WatermarkINN, compute_ber_loss, compute_accuracy


def pretrain_inn(opt):
    # 1. 准备数据
    device = torch.device('cuda')
    dataset = EndoDataset(opt, device=device, type='train')
    dataloader = dataset.dataloader()

    # 2. 初始化模型
    inn_model = WatermarkINN().cuda()
    optimizer = optim.Adam(inn_model.parameters(), lr=1e-3)

    # 3. 生成并固定密钥
    watermark_key = torch.randint(0, 2, (1, 64)).float().cuda()
    print(f"Generated Key: {watermark_key[0, :10].cpu().numpy()}...")

    # 4. 训练循环
    epochs = 50  # 纯图片训练很快，50轮足够收敛
    best_acc = 0.0

    os.makedirs(opt.workspace, exist_ok=True)

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        count = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for data in pbar:
            # 获取图片和Mask
            gt_image = data['camera'].original_image.cuda().unsqueeze(0)
            mask = data['mask'].cuda().unsqueeze(0).unsqueeze(0)

            # Forward: 嵌入 + 提取
            wm_image = inn_model.embed(gt_image, watermark_key)
            restored, w_logits = inn_model.extract(wm_image)

            # Loss 计算
            # 只在 Mask 区域保证画质，水印 Loss 则全局计算
            l_img = l1_loss(wm_image * mask, gt_image * mask)
            l_ber = compute_ber_loss(w_logits, watermark_key)
            loss = l_img + l_ber

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录指标
            acc = compute_accuracy(w_logits, watermark_key)
            epoch_loss += loss.item()
            epoch_acc += acc
            count += 1

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{acc:.4f}"})

        avg_acc = epoch_acc / count
        print(f"Epoch {epoch + 1} Average Acc: {avg_acc:.4f}")

        # 保存准确率最高的模型
        if avg_acc >= best_acc:
            best_acc = avg_acc
            torch.save(inn_model.state_dict(), os.path.join(opt.workspace, "best_inn.pth"))
            torch.save(watermark_key, os.path.join(opt.workspace, "watermark_key.pth"))
            print(f"Saved Best Model to {opt.workspace}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Dataset path")
    parser.add_argument('--workspace', type=str, default='output/inn_pretrain', help="Output folder for pretraining")
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1])
    opt = parser.parse_args()

    seed_everything(0)
    pretrain_inn(opt)