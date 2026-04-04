import torch
import torch.nn.functional as F
from utils.general_utils import build_rotation


class EndoGSWatermarker:
    def __init__(self, gaussian_model, secret_seed=42, amplitude=1e-5, msg_length=64):
        self.model = gaussian_model
        self.secret_seed = secret_seed
        self.amplitude = amplitude
        self.watermark_threshold = 0.7  # “高重要性”高斯的得分阈值

        # 多比特水印相关属性
        self.msg_length = msg_length  # 可控的二进制信息长度 (如 32, 48, 64)
        self.binary_msg = None  # 存储生成的二进制信息
        self.proj_matrix = None  # 扩频投影矩阵
        self.key = None

    def _get_or_generate_key(self):
        """生成人为固定的随机二进制信息，并通过投影矩阵转化为安全加性扰动密钥"""
        if self.key is None:
            torch.manual_seed(self.secret_seed)
            f_dim = self.model._features_rest.shape[1]
            feat_size = f_dim * 3

            # 1. 生成随机的二进制信息，并映射到 {-1, 1} 以增强正交性和解码稳定性
            # 形状: (msg_length,)
            self.binary_msg = torch.randint(0, 2, (self.msg_length,), device="cuda").float() * 2 - 1

            # 2. 生成正态分布的伪随机投影矩阵 (密钥载波)
            # 形状: (msg_length, feat_size)
            self.proj_matrix = torch.randn((self.msg_length, feat_size), device="cuda")

            # 3. 扩频变换：将二进制信息投影到高斯球谐特征空间
            # 形状: (feat_size,)
            raw_key = torch.matmul(self.binary_msg, self.proj_matrix)

            # 4. 能量约束：归一化并按照 amplitude 缩放，使其与原有逻辑的能量尺度严格一致
            raw_key = F.normalize(raw_key, dim=0) * self.amplitude * (feat_size ** 0.5)

            # 调整形状以满足广播机制: (1, F, 3)
            self.key = raw_key.view(1, f_dim, 3)

        return self.key

    def compute_importance_score(self):
        """计算三指标加权重要性分数"""
        # 1. 渲染贡献度 (不透明度 + 坐标梯度累积)
        opacity = self.model.get_opacity.detach().squeeze(-1)
        denom = self.model.denom.detach().squeeze(-1).clamp(min=1e-8)
        grad_accum = self.model.xyz_gradient_accum.detach().squeeze(-1) / denom

        norm_opacity = opacity / (opacity.max() + 1e-8)
        norm_grad = grad_accum / (grad_accum.max() + 1e-8)
        render_contrib = 0.5 * (norm_opacity + norm_grad)

        # 2. 空间分布 (缩放范数 + 2D最大半径)
        scaling = self.model.get_scaling.detach().norm(dim=1)
        radii = self.model.max_radii2D.detach()

        norm_scale = scaling / (scaling.max() + 1e-8)
        norm_radii = radii / (radii.max() + 1e-8)
        spatial_dist = 0.3 * (norm_scale + norm_radii) / 2.0

        # 3. 熵/密度特征 (1 - 不透明度熵值)
        p = opacity.clamp(1e-6, 1.0 - 1e-6)
        entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
        norm_entropy = entropy / (entropy.max() + 1e-8)
        density_feature = 0.2 * (1.0 - norm_entropy)

        score = render_contrib + spatial_dist + density_feature
        return score / (score.max() + 1e-8)

    def dynamic_prune_and_split(self, iteration):
        """在训练期间执行动态高斯精简与分裂调度"""
        scores = self.compute_importance_score()
        num_gaussians = scores.shape[0]

        if iteration < 20000:
            drop_rate, split_rate = 0.15, 0.10
        elif iteration < 40000:
            drop_rate, split_rate = 0.20, 0.05
        else:
            drop_rate, split_rate = 0.10, 0.00

        num_drop = int(num_gaussians * drop_rate)
        drop_thresh = torch.kthvalue(scores, max(1, num_drop)).values

        num_split = int(num_gaussians * split_rate)
        if num_split > 0:
            split_thresh = torch.kthvalue(scores, num_gaussians - num_split).values
        else:
            split_thresh = 2.0

        prune_mask = scores <= drop_thresh
        split_mask = scores >= split_thresh

        # 硬性约束：绝对不能丢弃（剪枝）高重要性的高斯
        prune_mask = prune_mask & (scores < self.watermark_threshold)

        # 1. 剪枝低重要性高斯
        self.model.prune_points(prune_mask)

        # 2. 分裂高重要性高斯
        valid_split_mask = split_mask[~prune_mask]
        self._force_split(valid_split_mask)

    def _force_split(self, split_mask, N=2):
        """将高斯点分裂为 N 个子点，并完美继承球谐特征"""
        if not split_mask.any(): return

        stds = self.model.get_scaling[split_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.model._rotation[split_mask]).repeat(N, 1, 1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.model.get_xyz[split_mask].repeat(N, 1)
        new_scaling = self.model.scaling_inverse_activation(self.model.get_scaling[split_mask].repeat(N, 1) / (0.8 * N))

        new_rotation = self.model._rotation[split_mask].repeat(N, 1)
        new_features_dc = self.model._features_dc[split_mask].repeat(N, 1, 1)
        new_features_rest = self.model._features_rest[split_mask].repeat(N, 1, 1)
        new_opacity = self.model._opacity[split_mask].repeat(N, 1)

        # 兼容可能有或没有 deformation_table 的情况
        if hasattr(self.model, '_deformation_table') and self.model._deformation_table is not None:
            new_deformation_table = self.model._deformation_table[split_mask].repeat(N)
            self.model.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling,
                                             new_rotation, new_deformation_table)
        else:
            self.model.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling,
                                             new_rotation)

        prune_filter = torch.cat((split_mask, torch.zeros(N * split_mask.sum(), device="cuda", dtype=torch.bool)))
        self.model.prune_points(prune_filter)

    def embed_watermark(self):
        """训练后阶段：向高重要性高斯的球谐特征中加入安全扰动"""
        key = self._get_or_generate_key()
        scores = self.compute_importance_score()
        mask = scores >= self.watermark_threshold

        with torch.no_grad():
            self.model._features_rest[mask] += key

        print(f"[水印系统] 成功将版权水印安全嵌入至 {mask.sum().item()} 个高重要性高斯点中。")

    def extract_and_verify(self):
        """推理阶段：不仅进行 ZNCC 匹配，同时提取并验证嵌入的二进制信息"""
        key = self._get_or_generate_key()
        scores = self.compute_importance_score()
        mask = scores >= self.watermark_threshold

        features = self.model._features_rest[mask].detach()

        # 1. 传统存在性验证
        key_expanded = key.expand_as(features)
        feat_c = features - features.mean(dim=(1, 2), keepdim=True)
        key_c = key_expanded - key_expanded.mean(dim=(1, 2), keepdim=True)

        correlation = (feat_c * key_c).sum(dim=(1, 2)) / (
                torch.norm(feat_c, dim=(1, 2)) * torch.norm(key_c, dim=(1, 2)) + 1e-8)

        match_rate = (correlation > 0.5).float().mean().item()
        is_authentic = match_rate >= 0.95

        # 2. 多比特信息解码
        mean_watermarked_features = features.mean(dim=0).view(-1)
        decoded_msg_soft = torch.matmul(self.proj_matrix, mean_watermarked_features)
        decoded_msg = torch.sign(decoded_msg_soft)

        bit_accuracy = (decoded_msg == self.binary_msg).float().mean().item()

        print(f"[水印系统] 版权验证完成。授权状态: {is_authentic} | 宏观匹配率: {match_rate:.2%}")
        print(
            f"[水印系统] 载荷提取完成。比特准确率: {bit_accuracy:.2%} ({int(bit_accuracy * self.msg_length)}/{self.msg_length} bits)")

        return is_authentic, match_rate, bit_accuracy

    def recover_model(self):
        """推理阶段：减去精确的扰动值，实现 100% 原始参数的无损恢复"""
        key = self._get_or_generate_key()
        scores = self.compute_importance_score()
        mask = scores >= self.watermark_threshold

        with torch.no_grad():
            self.model._features_rest[mask] -= key

        print("[水印系统] 模型参数已完美恢复至无损原始状态。")