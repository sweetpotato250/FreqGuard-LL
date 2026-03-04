import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import numpy as np
import io
import copy
from PIL import Image


class RobustnessAttacker:
    """
    针对 3DGS 水印的专用攻击库。
    不仅包含 2D 图像域攻击，还包含 3D 参数域和视角域的特有攻击。
    """

    # ==========================================================================
    # 2D 图像域攻击 (Image-level Attacks)
    # ==========================================================================
    @staticmethod
    def attack_noise(image, std=0.1):
        noise = torch.randn_like(image) * std
        return torch.clamp(image + noise, 0, 1)

    @staticmethod
    def attack_rotation(image, angle_deg=30):
        return TF.rotate(image, angle_deg, interpolation=TF.InterpolationMode.BILINEAR)

    @staticmethod
    def attack_scaling(image, scale=0.25):
        B, C, H, W = image.shape
        new_H, new_W = int(H * scale), int(W * scale)
        downscaled = F.interpolate(image, size=(new_H, new_W), mode='bilinear', align_corners=False)
        recovered = F.interpolate(downscaled, size=(H, W), mode='bilinear', align_corners=False)
        return recovered

    @staticmethod
    def attack_blur(image, sigma=1.0, kernel_size=5):
        if kernel_size % 2 == 0: kernel_size += 1
        blur_transform = T.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=sigma)
        return blur_transform(image)

    @staticmethod
    def attack_crop(image, crop_percent=0.4):
        B, C, H, W = image.shape
        keep_ratio = 1.0 - crop_percent
        crop_h, crop_w = int(H * keep_ratio), int(W * keep_ratio)
        cropped = TF.center_crop(image, [crop_h, crop_w])
        resized = F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
        return resized

    @staticmethod
    def attack_brightness(image, factor=2.0):
        return torch.clamp(image * factor, 0, 1)

    @staticmethod
    def attack_jpeg(image, quality=10):
        img_cpu = image.detach().cpu()
        processed_imgs = []
        for i in range(img_cpu.shape[0]):
            tensor_img = img_cpu[i]
            pil_img = TF.to_pil_image(tensor_img.clamp(0, 1))
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            jpeg_img = Image.open(buffer)
            tensor_out = TF.to_tensor(jpeg_img)
            processed_imgs.append(tensor_out)
        return torch.stack(processed_imgs).to(image.device)

    @staticmethod
    def attack_combined(image):
        x = RobustnessAttacker.attack_crop(image, crop_percent=0.4)
        x = RobustnessAttacker.attack_brightness(x, factor=2.0)
        x = RobustnessAttacker.attack_jpeg(x, quality=10)
        return x

    # ==========================================================================
    # 3D 参数域与视角域攻击 (3D-Specific Attacks) - 解决缺陷二
    # ==========================================================================

    @staticmethod
    def apply_pruning(gaussian_model, prune_percent=0.1):
        """
        [3D攻击] 随机裁剪(丢弃)指定比例的高斯点。
        返回原始的 opacity 参数，用于攻击测试后恢复模型。
        """
        original_opacity = gaussian_model._opacity.clone().detach()
        # 生成随机掩码，决定哪些点被保留
        mask = torch.rand(original_opacity.shape, device=original_opacity.device) > prune_percent
        # 将被丢弃的点的不透明度设为0 (极小值避免数学错误)
        gaussian_model._opacity.data = original_opacity * mask.float() - 9999.0 * (~mask).float()
        return original_opacity

    @staticmethod
    def apply_xyz_noise(gaussian_model, std=0.005):
        """
        [3D攻击] 给高斯点的三维空间坐标(XYZ)添加噪声，模拟点云扰动。
        返回原始的 XYZ 参数，用于恢复。
        """
        original_xyz = gaussian_model._xyz.clone().detach()
        noise = torch.randn_like(original_xyz) * std
        gaussian_model._xyz.data = original_xyz + noise
        return original_xyz

    @staticmethod
    def restore_attribute(gaussian_model, attr_name, original_data):
        """通用 3D 属性恢复函数"""
        getattr(gaussian_model, attr_name).data = original_data

    @staticmethod
    def perturb_camera(camera, trans_std=0.05):
        """
        [3D攻击] 视角微调攻击 (Viewpoint Change)
        给相机的世界坐标系变换添加微小偏移，模拟测试视角与原视角不完全一致。
        """
        new_cam = copy.copy(camera)
        # 深拷贝需要被修改的矩阵
        new_cam.world_view_transform = camera.world_view_transform.clone()

        # 对平移向量添加噪声 (位于矩阵第 4 行，前 3 列)
        noise = torch.randn(3, device=new_cam.world_view_transform.device) * trans_std
        new_cam.world_view_transform[3, :3] += noise

        # 重新计算投影矩阵和相机中心
        new_cam.full_proj_transform = new_cam.world_view_transform.unsqueeze(0).bmm(
            new_cam.projection_matrix.unsqueeze(0)).squeeze(0)
        new_cam.camera_center = new_cam.world_view_transform.inverse()[3, :3]
        return new_cam