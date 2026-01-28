import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# Part 1: DWT & IDWT (小波变换)
# ==========================================
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        # x: [B, C, H, W]
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return torch.cat([x_LL, x_HL, x_LH, x_HH], dim=1)

class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        in_batch, in_channel, in_height, in_width = x.size()
        out_channel = int(in_channel / 4)
        out_height = 2 * in_height
        out_width = 2 * in_width
        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

        h = torch.zeros([in_batch, out_channel, out_height, out_width]).float().to(x.device)
        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
        return h

# ==========================================
# Part 2: INN Block (可逆网络层)
# ==========================================
class CouplingLayer(nn.Module):
    def __init__(self, in_channels, mid_channels=64):
        super(CouplingLayer, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels // 2, mid_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_channels // 2, 3, 1, 1)
        )

    def forward(self, x, reverse=False):
        x1, x2 = torch.chunk(x, 2, dim=1)
        if not reverse:
            y1 = x1
            y2 = x2 + self.net(x1)
            return torch.cat([y1, y2], dim=1)
        else:
            y1 = x1
            y2 = x2 - self.net(x1)
            return torch.cat([y1, y2], dim=1)

class WatermarkINN(nn.Module):
    def __init__(self, in_channels=12, steps=3, wm_len=64):
        super(WatermarkINN, self).__init__()
        self.dwt = DWT()
        self.idwt = IDWT()
        self.wm_len = wm_len
        self.layers = nn.ModuleList([CouplingLayer(in_channels) for _ in range(steps)])
        self.wm_projection = nn.Linear(wm_len, in_channels)
        self.wm_extract_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, wm_len)
        )

    def embed(self, image, watermark):
        """将水印嵌入图像 (Forward)"""
        coeffs = self.dwt(image)
        wm_feat = self.wm_projection(watermark).unsqueeze(-1).unsqueeze(-1)
        x = coeffs + wm_feat
        for layer in self.layers:
            x = layer(x, reverse=False)
        return self.idwt(x)

    def extract(self, watermarked_image):
        """从图像提取水印 (Reverse)"""
        coeffs = self.dwt(watermarked_image)
        x = coeffs
        for layer in reversed(self.layers):
            x = layer(x, reverse=True)
        w_pred_logits = self.wm_extract_head(x)
        restored_image = self.idwt(x)
        return restored_image, w_pred_logits

# ==========================================
# Part 3: Utility Functions (损失与精度)
# ==========================================
def compute_ber_loss(w_pred_logits, w_gt):
    """计算比特错误率损失 (BCE)"""
    return F.binary_cross_entropy_with_logits(w_pred_logits, w_gt)

def compute_accuracy(w_pred_logits, w_gt):
    """计算比特匹配准确率 (0.0 - 1.0)"""
    w_pred_bits = (torch.sigmoid(w_pred_logits) > 0.5).float()
    correct_bits = (w_pred_bits == w_gt).float()
    return correct_bits.mean().item()