import torch
import torch.nn as nn
import torch.nn.functional as F
from DTC import DTC


# --- 1. 提取自 resnet_cbam.py 的 CBAM 模块 ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# --- 2. 结合 CBAM 的双重卷积块 ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_cbam=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.use_cbam = use_cbam
        if use_cbam:
            self.ca = ChannelAttention(out_channels)
            self.sa = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        if self.use_cbam:
            x = self.ca(x) * x
            x = self.sa(x) * x
        return x


# --- 3. 反向注意力模块 (Reverse Attention) ---
class ReverseAttention(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super(ReverseAttention, self).__init__()
        self.conv_feat = DoubleConv(in_channels, out_channels, use_cbam=True)
        self.conv_pred = nn.Conv2d(out_channels, 1, kernel_size=3, padding=1)

    def forward(self, x, coarse_mask):
        # 强制对齐粗糙掩码与当前特征图的尺寸
        if coarse_mask.shape[2:] != x.shape[2:]:
            coarse_mask = F.interpolate(coarse_mask, size=x.shape[2:], mode='bilinear', align_corners=False)

        # 核心：反转掩码，迫使模型关注边界和错误区域
        reverse_weight = 1.0 - torch.sigmoid(coarse_mask)
        x_refined = self.conv_feat(x * reverse_weight)
        fine_mask = self.conv_pred(x_refined)

        return x_refined, fine_mask


# --- 4. 完整的网络架构 ---
class UNet_DTC_PraNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # 编码器
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        # 解码阶段 1
        self.up1 = DTC(in_channels=512, out_channels=256, scale=2, dim=2)
        self.conv_up1 = DoubleConv(512, 256)
        self.pred_head_coarse = nn.Conv2d(256, out_channels, kernel_size=3, padding=1)

        # 解码阶段 2 + RA
        self.up2 = DTC(in_channels=256, out_channels=128, scale=2, dim=2)
        self.conv_up2 = DoubleConv(256, 128)
        self.ra2 = ReverseAttention(in_channels=128, out_channels=64)

        # 解码阶段 3 + RA
        self.up3 = DTC(in_channels=64, out_channels=64, scale=2, dim=2)
        self.conv_up3 = DoubleConv(128, 64)
        self.ra3 = ReverseAttention(in_channels=64, out_channels=32)

    def forward(self, x):
        input_size = x.shape[2:]

        # 编码
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        b = self.down4(self.pool(x3))

        # 解码 1
        d1 = self.up1(b)
        # 【尺寸修复点】强制将 d1 缩放对齐 x3
        if d1.shape[2:] != x3.shape[2:]:
            d1 = F.interpolate(d1, size=x3.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, x3], dim=1)
        d1 = self.conv_up1(d1)
        pred_coarse = self.pred_head_coarse(d1)

        # 解码 2
        d2 = self.up2(d1)
        # 【尺寸修复点】强制将 d2 缩放对齐 x2
        if d2.shape[2:] != x2.shape[2:]:
            d2 = F.interpolate(d2, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.conv_up2(d2)
        feat_ra2, pred_mid = self.ra2(d2, pred_coarse)

        # 解码 3
        d3 = self.up3(feat_ra2)
        # 【尺寸修复点】强制将 d3 缩放对齐 x1
        if d3.shape[2:] != x1.shape[2:]:
            d3 = F.interpolate(d3, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, x1], dim=1)
        d3 = self.conv_up3(d3)
        _, pred_fine = self.ra3(d3, pred_mid)

        # 【尺寸修复点】将所有预测插值回最初始的输入尺寸，防止 Loss 报错
        pred_coarse = F.interpolate(pred_coarse, size=input_size, mode='bilinear', align_corners=False)
        pred_mid = F.interpolate(pred_mid, size=input_size, mode='bilinear', align_corners=False)
        pred_fine = F.interpolate(pred_fine, size=input_size, mode='bilinear', align_corners=False)

        if self.training:
            return pred_fine, pred_mid, pred_coarse
        else:
            return pred_fine