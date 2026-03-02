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
    针对 3DGS 医学影像（临床/传输场景）的专用水印鲁棒性攻击库。
    参数严格对应医疗场景要求：
    1. 信号噪声 (Gaussian Noise): 扫描/传输噪声，σ=0.001~0.01
    2. 顶点简化 (Vertex Simplification): 降低模型分辨率，简化率 10%~30%
    3. 平滑滤波 (Smooth Filtering): 去除扫描噪声，核大小 3x3，迭代 1~3 次
    4. 相似变换 (Similarity Transform): 不同设备对齐，旋转 ±15°，平移 ±5mm
    5. 格式转换 (Format Conversion): 格式互转导致精度丢失 (截断模拟)
    6. 轻微裁剪 (Mild Cropping): 避开器官主体，裁剪面积 < 10%
    """

    # ==========================================================================
    # 1. 扫描/传输信号噪声 (Gaussian Noise)
    # 医疗扫描精度级：σ = 0.001 ~ 0.01
    # ==========================================================================
    @staticmethod
    def attack_noise_2d(image, std=0.005):
        """对 2D 渲染切片添加微弱信号噪声 (默认 σ=0.005)"""
        noise = torch.randn_like(image) * std
        return torch.clamp(image + noise, 0, 1)

    @staticmethod
    def attack_noise_3d(gaussian_model, std=0.005):
        """对 3DGS 模型的空间坐标(XYZ)添加微量扫描噪声"""
        original_xyz = gaussian_model._xyz.clone()
        noise = torch.randn_like(gaussian_model._xyz) * std
        gaussian_model._xyz.data += noise
        return original_xyz  # 返回原坐标用于后续恢复

    # ==========================================================================
    # 2. 临床预览顶点简化 / 下采样 (Vertex Simplification)
    # 保留关键解剖结构：简化率 10% ~ 30%
    # ==========================================================================
    @staticmethod
    def get_pruning_mask(gaussian_model, prune_percent=0.20):
        """
        基于不透明度 (Opacity) 剪枝，模拟临床 3D 预览模型下采样。
        默认 prune_percent=0.20 (剔除 20% 最不重要的点)
        """
        opacities = gaussian_model._opacity.detach().squeeze()
        k = int(opacities.shape[0] * prune_percent)
        if k == 0:
            return torch.ones_like(opacities, dtype=torch.bool)
        try:
            threshold, _ = torch.kthvalue(opacities, k)
        except RuntimeError:
            threshold, _ = torch.kthvalue(opacities.cpu(), k)
            threshold = threshold.to(opacities.device)
        return opacities > threshold

    @staticmethod
    def apply_temp_pruning(gaussian_model, mask):
        original_opacity = gaussian_model._opacity.clone()
        gaussian_model._opacity.data[~mask] = 0.0
        return original_opacity

    # ==========================================================================
    # 3. 平滑滤波 (Smooth Filtering)
    # 去除扫描噪声：高斯平滑核大小 3×3，迭代 1~3 次
    # ==========================================================================
    @staticmethod
    def attack_smooth_2d(image, kernel_size=3, iterations=2):
        """
        模拟医疗影像后处理中的常规平滑去噪。
        采用较小的高斯核 (3x3)，以保留解剖边缘。
        """
        sigma = 0.5
        blur_transform = T.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=sigma)

        smoothed = image
        for _ in range(iterations):
            smoothed = blur_transform(smoothed)
        return smoothed

    # ==========================================================================
    # 4. 相似变换 (Similarity Transform: Rotation / Translation)
    # 不同设备/视角对齐：旋转 ±15°，平移 ±5mm (假设 1 坐标单位=1米，5mm=0.005)
    # ==========================================================================
    @staticmethod
    def attack_transform_3d(gaussian_model, angle_deg=15.0, trans_units=0.005):
        """对 3DGS 模型本身进行刚体变换 (此处以绕 Y 轴旋转为例)"""
        original_xyz = gaussian_model._xyz.clone()

        # 构造旋转矩阵
        theta = np.radians(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot_mat = torch.tensor([
            [cos_t, 0, sin_t],
            [0, 1, 0],
            [-sin_t, 0, cos_t]
        ], device=gaussian_model._xyz.device, dtype=gaussian_model._xyz.dtype)

        # 应用旋转与平移
        rotated_xyz = torch.matmul(gaussian_model._xyz.data, rot_mat.T)
        translated_xyz = rotated_xyz + trans_units

        gaussian_model._xyz.data = translated_xyz
        return original_xyz

    # ==========================================================================
    # 5. 格式转换 (Format Conversion)
    # PLY -> OBJ -> 3DGS 转换：模拟精度截断与重组
    # ==========================================================================
    @staticmethod
    def attack_format_conversion_3d(gaussian_model):
        """
        通过 ASCII 格式（如 OBJ）互转时，浮点数往往被截断到 4 位小数。
        这里模拟底层坐标与 DC 属性的极微小精度丢失（量化误差）。
        """
        original_xyz = gaussian_model._xyz.clone()
        original_features = gaussian_model._features_dc.clone()

        # 模拟浮点数截断 (Float32 -> ASCII 4位小数 -> Float32)
        gaussian_model._xyz.data = torch.round(gaussian_model._xyz.data * 10000) / 10000
        gaussian_model._features_dc.data = torch.round(gaussian_model._features_dc.data * 10000) / 10000

        return original_xyz, original_features

    # ==========================================================================
    # 6. 轻微裁剪 (Mild Cropping)
    # 裁剪非关键背景：面积 < 10% (避开器官主体)
    # ==========================================================================
    @staticmethod
    def attack_mild_crop_2d(image, crop_percent=0.05):
        """
        crop_percent: 默认 0.05 (切掉 5%，保留中心 95%)。
        避免像以前 40% 那样破坏主要解剖结构。
        """
        B, C, H, W = image.shape
        keep_ratio = 1.0 - crop_percent

        crop_h = int(H * keep_ratio)
        crop_w = int(W * keep_ratio)

        # 居中裁剪并拉伸回原尺寸
        cropped = TF.center_crop(image, [crop_h, crop_w])
        resized = F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
        return resized

    # ==========================================================================
    # 通用恢复函数 (Restore Modules)
    # 必须在每次测试完 3D 攻击后调用，以避免影响下一次测试
    # ==========================================================================
    @staticmethod
    def restore_model(gaussian_model, original_opacity):
        """恢复透明度 (供剪枝测试后调用)"""
        gaussian_model._opacity.data = original_opacity

    @staticmethod
    def restore_model_xyz(gaussian_model, original_xyz):
        """恢复 3D 坐标 (供平移、旋转、噪声测试后调用)"""
        gaussian_model._xyz.data = original_xyz

    @staticmethod
    def restore_model_features(gaussian_model, original_features):
        """恢复颜色/特征 (供格式转换测试后调用)"""
        gaussian_model._features_dc.data = original_features