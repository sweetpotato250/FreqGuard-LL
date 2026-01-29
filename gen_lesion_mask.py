import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from PIL import Image
from torchvision import transforms
import cv2

# 假设你已经有了 PraNet 的模型定义文件 (PraNet_Res2Net.py)
from PraNet_Res2Net import PraNet


def generate_masks(model_path, data_path, save_path):
    # 1. 加载 PraNet 模型
    model = PraNet().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    os.makedirs(save_path, exist_ok=True)

    # 2. 遍历你的 EndoGS 图片
    # EndoGS 图片通常在 data_path/images 目录下
    img_folder = os.path.join(data_path, "images")
    image_list = os.listdir(img_folder)

    transform = transforms.Compose([
        transforms.Resize((352, 352)),  # PraNet 标准输入尺寸
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print(f"Start generating masks for {len(image_list)} images...")

    for image_name in image_list:
        if not image_name.endswith(('.jpg', '.png')): continue

        # 读取图片
        image_path = os.path.join(img_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        # 记录原始尺寸，以便最后还原
        orig_w, orig_h = image.size

        # 预处理
        images = transform(image).unsqueeze(0).cuda()

        # 推理
        with torch.no_grad():
            res5, res4, res3, res2 = model(images)
            # res2 是最终输出
            res = res2
            # Sigmoid 归一化
            res = F.upsample(res, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()

        # 二值化 (0.5 为阈值)
        # res > 0.5 是病变，<= 0.5 是背景
        mask = (res - res.min()) / (res.max() - res.min() + 1e-8)
        mask = (mask > 0.5).astype(np.uint8) * 255

        # 保存掩码 (名字要和原图对应，例如 frame001.png)
        save_name = os.path.join(save_path, image_name.replace(".jpg", ".png"))
        cv2.imwrite(save_name, mask)

    print(f"Done! Lesion masks saved to {save_path}")


if __name__ == '__main__':
    # 修改这里的路径
    dataset_root = "../data/endonerf/cutting_tissues_twice"  # 你的 EndoGS 数据集路径
    model_ckpt = "snapshots/PraNet_Res2Net.pth"  # 你下载的权重路径
    output_folder = os.path.join(dataset_root, "lesion_masks")  # 输出文件夹

    generate_masks(model_ckpt, dataset_root, output_folder)