import torch
import torch.nn.functional as F
from utils.general_utils import build_rotation


class EndoGSWatermarker:
    def __init__(self, gaussian_model, secret_seed=42, amplitude=1e-5):
        self.model = gaussian_model
        self.secret_seed = secret_seed
        self.amplitude = amplitude
        self.key = None
        self.watermark_threshold = 0.7  # “高重要性”高斯的得分阈值

    def _get_or_generate_key(self):
        """生成用于 _features_rest 的安全加性扰动密钥"""
        if self.key is None:
            torch.manual_seed(self.secret_seed)
            # _features_rest 的形状为: (N, F, 3)。密钥形状为: (1, F, 3) 以便广播
            f_dim = self.model._features_rest.shape[1]
            self.key = torch.randn((1, f_dim, 3), device="cuda") * self.amplitude
        return self.key

    def compute_importance_score(self):
        """
        计算三指标加权重要性分数。
        分数 = 0.5 * 渲染贡献度 + 0.3 * 空间分布 + 0.2 * 熵/密度特征
        """
        # 1. 渲染贡献度 (不透明度 + 坐标梯度累积)
        opacity = self.model.get_opacity.detach().squeeze(-1)
        denom = self.model.denom.detach().squeeze(-1).clamp(min=1e-8)
        grad_accum = self.model.xyz_gradient_accum.detach().squeeze(-1) / denom

        norm_opacity = opacity / (opacity.max() + 1e-8)
        norm_grad = grad_accum / (grad_accum.max() + 1e-8)
        render_contrib = 0.5 * (norm_opacity + norm_grad)  # 最大权重 0.5

        # 2. 空间分布 (缩放范数 + 2D最大半径)
        scaling = self.model.get_scaling.detach().norm(dim=1)
        radii = self.model.max_radii2D.detach()

        norm_scale = scaling / (scaling.max() + 1e-8)
        norm_radii = radii / (radii.max() + 1e-8)
        spatial_dist = 0.3 * (norm_scale + norm_radii) / 2.0  # 最大权重 0.3

        # 3. 熵/密度特征 (1 - 不透明度熵值)
        p = opacity.clamp(1e-6, 1.0 - 1e-6)
        entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
        norm_entropy = entropy / (entropy.max() + 1e-8)
        density_feature = 0.2 * (1.0 - norm_entropy)  # 最大权重 0.2

        score = render_contrib + spatial_dist + density_feature
        # 归一化到 [0, 1] 范围内，以确保阈值稳定
        return score / (score.max() + 1e-8)

    def dynamic_prune_and_split(self, iteration):
        """在训练期间执行动态高斯精简与分裂调度"""
        scores = self.compute_importance_score()
        num_gaussians = scores.shape[0]

        # 匹配方案设计中的三个阶段
        if iteration < 20000:
            drop_rate, split_rate = 0.15, 0.10
        elif iteration < 40000:
            drop_rate, split_rate = 0.20, 0.05
        else:
            drop_rate, split_rate = 0.10, 0.00

        # 根据指定的比例计算动态阈值
        num_drop = int(num_gaussians * drop_rate)
        drop_thresh = torch.kthvalue(scores, max(1, num_drop)).values

        num_split = int(num_gaussians * split_rate)
        if num_split > 0:
            split_thresh = torch.kthvalue(scores, num_gaussians - num_split).values
        else:
            split_thresh = 2.0  # 设置一个不可能达到的阈值以阻止分裂

        prune_mask = scores <= drop_thresh
        split_mask = scores >= split_thresh

        # 硬性约束：绝对不能丢弃（剪枝）高重要性的高斯
        prune_mask = prune_mask & (scores < self.watermark_threshold)

        # 1. 剪枝低重要性高斯
        self.model.prune_points(prune_mask)

        # 2. 分裂高重要性高斯（必须滤除刚才已被剪枝的点，避免索引越界）
        valid_split_mask = split_mask[~prune_mask]
        self._force_split(valid_split_mask)

    def _force_split(self, split_mask, N=2):
        """将高斯点分裂为 N 个子点，并完美继承球谐特征，不依赖梯度"""
        if not split_mask.any(): return

        stds = self.model.get_scaling[split_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.model._rotation[split_mask]).repeat(N, 1, 1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.model.get_xyz[split_mask].repeat(N, 1)
        new_scaling = self.model.scaling_inverse_activation(self.model.get_scaling[split_mask].repeat(N, 1) / (0.8 * N))

        # 完美继承所有属性，特别是必须原封不动继承 _features_rest
        new_rotation = self.model._rotation[split_mask].repeat(N, 1)
        new_features_dc = self.model._features_dc[split_mask].repeat(N, 1, 1)
        new_features_rest = self.model._features_rest[split_mask].repeat(N, 1, 1)
        new_opacity = self.model._opacity[split_mask].repeat(N, 1)
        new_deformation_table = self.model._deformation_table[split_mask].repeat(N)

        self.model.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling,
                                         new_rotation, new_deformation_table)

        # 剪枝原始的父代高斯点
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
        """推理阶段：提取特征，与密钥进行交叉相关计算，验证版权"""
        key = self._get_or_generate_key()
        scores = self.compute_importance_score()
        mask = scores >= self.watermark_threshold

        features = self.model._features_rest[mask].detach()
        key_expanded = key.expand_as(features)

        # 零均值归一化互相关 (ZNCC)
        feat_c = features - features.mean(dim=(1, 2), keepdim=True)
        key_c = key_expanded - key_expanded.mean(dim=(1, 2), keepdim=True)

        correlation = (feat_c * key_c).sum(dim=(1, 2)) / (
                    torch.norm(feat_c, dim=(1, 2)) * torch.norm(key_c, dim=(1, 2)) + 1e-8)

        # 计算所有目标高斯点的匹配率
        match_rate = (correlation > 0.5).float().mean().item()
        is_authentic = match_rate >= 0.95

        print(f"[水印系统] 版权验证完成。授权状态: {is_authentic} | 匹配率: {match_rate:.2%}")
        return is_authentic, match_rate

    def recover_model(self):
        """推理阶段：减去精确的扰动值，实现 100% 原始参数的无损恢复"""
        key = self._get_or_generate_key()
        # 由于我们只修改了 _features_rest，不透明度/缩放/坐标完全未变。
        # 因此，此处动态计算的得分能完美匹配嵌入时的掩码 (mask)。
        scores = self.compute_importance_score()
        mask = scores >= self.watermark_threshold

        with torch.no_grad():
            self.model._features_rest[mask] -= key

        print("[水印系统] 模型参数已完美恢复至无损原始状态。")