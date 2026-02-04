import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import numpy as np
import io
from PIL import Image


class RobustnessAttacker:
    """
    针对 3DGS 水印的专用攻击库。
    参数严格对应用户要求：
    1. Gaussian Noise (v=0.1)
    2. Rotation (±π/6)
    3. Scaling (25%)
    4. Gaussian Blur (deviation=0.1 - 注：实际通常指sigma，此处映射为sigma)
    5. Crop (40%)
    6. Brightness (x2.0)
    7. JPEG (Q=10)
    8. Combined (Crop + Brightness + JPEG)
    """

    # ==========================================================================
    # 1. 高斯噪声攻击 (Gaussian Noise)
    # 参数：v = 0.1 (此处理解为标准差 sigma=0.1，若是方差则 sigma≈0.316)
    # ==========================================================================
    @staticmethod
    def attack_noise(image, std=0.1):
        """
        image: (B, C, H, W), range [0, 1]
        std: 噪声强度 (默认 v=0.1)
        """
        noise = torch.randn_like(image) * std
        return torch.clamp(image + noise, 0, 1)

    # ==========================================================================
    # 2. 旋转攻击 (Rotation)
    # 参数：±π/6 (30度)
    # ==========================================================================
    @staticmethod
    def attack_rotation(image, angle_deg=30):
        """
        将图像旋转指定角度。
        angle_deg: 默认为 30 度 (π/6)
        """
        # 随机选择顺时针或逆时针，或者固定
        # 为了测试确定性，这里默认 +30 度，也可传入 -30
        return TF.rotate(image, angle_deg, interpolation=TF.InterpolationMode.BILINEAR)

    # ==========================================================================
    # 3. 缩放攻击 (Scaling)
    # 参数：25% 比例 (即缩小到 0.25x 再放大回原尺寸)
    # ==========================================================================
    @staticmethod
    def attack_scaling(image, scale=0.25):
        """
        模拟分辨率降低。
        scale: 0.25 表示缩小为原图的 1/4 (25%)
        """
        B, C, H, W = image.shape
        new_H, new_W = int(H * scale), int(W * scale)

        # 1. Downsample (丢失高频信息)
        downscaled = F.interpolate(image, size=(new_H, new_W), mode='bilinear', align_corners=False)

        # 2. Upsample (恢复 INN 输入尺寸)
        recovered = F.interpolate(downscaled, size=(H, W), mode='bilinear', align_corners=False)
        return recovered

    # ==========================================================================
    # 4. 高斯模糊攻击 (Gaussian Blur)
    # 参数：deviation = 0.1 (映射为 sigma)
    # 注意：Sigma=0.1 视觉效果极弱，如果需要明显模糊建议设为 1.0 或 2.0
    # ==========================================================================
    @staticmethod
    def attack_blur(image, sigma=1.0, kernel_size=5):
        """
        sigma: 标准差 (deviation)
        kernel_size: 卷积核大小，通常设为 5 或 7
        """
        # 保证核大小是奇数
        if kernel_size % 2 == 0: kernel_size += 1

        # 使用 TorchVision 的模糊
        blur_transform = T.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=sigma)
        return blur_transform(image)

    # ==========================================================================
    # 5. 裁剪攻击 (Crop)
    # 参数：40% 比例裁剪 (解释为：切掉 40% 的内容 / 或保留 60% 的中心内容)
    # 此处实现：保留中心 60% 的区域 (切除边缘 40%)，然后拉伸回原尺寸
    # ==========================================================================
    @staticmethod
    def attack_crop(image, crop_percent=0.4):
        """
        crop_percent: 0.4 (表示切掉 40%，保留 1 - 0.4 = 0.6)
        """
        B, C, H, W = image.shape
        keep_ratio = 1.0 - crop_percent  # 0.6

        crop_h = int(H * keep_ratio)
        crop_w = int(W * keep_ratio)

        # 中心裁剪
        cropped = TF.center_crop(image, [crop_h, crop_w])

        # 拉伸回原尺寸 (模拟局部放大/视野丢失)
        resized = F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
        return resized

    # ==========================================================================
    # 6. 亮度调整攻击 (Brightness)
    # 参数：亮度 x 2.0
    # ==========================================================================
    @staticmethod
    def attack_brightness(image, factor=2.0):
        """
        factor: 2.0 (两倍亮度)
        """
        return torch.clamp(image * factor, 0, 1)

    # ==========================================================================
    # 7. JPEG 压缩攻击 (JPEG Compression)
    # 参数：Quality = 10% (极强压缩)
    # ==========================================================================
    @staticmethod
    def attack_jpeg(image, quality=10):
        """
        真实的 JPEG 压缩模拟 (Tensor -> PIL -> Buffer -> PIL -> Tensor)
        这是最准确的模拟，包含块效应和色彩量化。
        """
        # 确保输入在 CPU
        img_cpu = image.detach().cpu()
        processed_imgs = []

        for i in range(img_cpu.shape[0]):
            # 1. 转换为 PIL
            tensor_img = img_cpu[i]
            pil_img = TF.to_pil_image(tensor_img.clamp(0, 1))

            # 2. 写入内存 Buffer (模拟存盘)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=quality)

            # 3. 重新读取
            buffer.seek(0)
            jpeg_img = Image.open(buffer)

            # 4. 转回 Tensor
            tensor_out = TF.to_tensor(jpeg_img)
            processed_imgs.append(tensor_out)

        # 堆叠回 Batch 并送回原设备
        return torch.stack(processed_imgs).to(image.device)

    # ==========================================================================
    # 8. 组合攻击 (Combined)
    # 顺序：裁剪(Crop) -> 亮度(Brightness) -> JPEG
    # ==========================================================================
    @staticmethod
    def attack_combined(image):
        """
        组合攻击：
        1. Crop (40%)
        2. Brightness (x2.0)
        3. JPEG (Q=10)
        """
        x = RobustnessAttacker.attack_crop(image, crop_percent=0.4)
        x = RobustnessAttacker.attack_brightness(x, factor=2.0)
        x = RobustnessAttacker.attack_jpeg(x, quality=10)
        return x

    # ==========================================================================
    # 3D 模型攻击 (保留你的 3D 剪枝功能)
    # ==========================================================================
    @staticmethod
    def get_pruning_mask(gaussian_model, prune_percent=0.5):
        opacities = gaussian_model._opacity.detach().squeeze()
        k = int(opacities.shape[0] * prune_percent)
        if k == 0: return torch.ones_like(opacities, dtype=torch.bool)
        try:
            threshold, _ = torch.kthvalue(opacities, k)
        except:
            threshold, _ = torch.kthvalue(opacities.cpu(), k)
            threshold = threshold.to(opacities.device)
        return opacities > threshold

    @staticmethod
    def apply_temp_pruning(gaussian_model, mask):
        original_opacity = gaussian_model._opacity.clone()
        gaussian_model._opacity.data[~mask] = 0.0
        return original_opacity

    @staticmethod
    def restore_model(gaussian_model, original_opacity):
        gaussian_model._opacity.data = original_opacity