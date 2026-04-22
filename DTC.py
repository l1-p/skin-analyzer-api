import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 论文：DTC: A Deformable Transposed Convolution Module for MedicalImage Segmentation
# 论文地址：https://arxiv.org/pdf/2601.17939
class DTC(nn.Module):
    """
    可变形转置卷积模块(Deformable Transposed Convolution)
    支持2D/3D医学图像分割，适配线性插值/转置卷积作为原始上采样方法
    兼容PyTorch 1.13+所有版本，修复grid_sample模式参数校验问题
    Args:
        in_channels: 输入特征通道数
        out_channels: 输出特征通道数
        scale: 上采样尺度 (int, 如2/4)
        dim: 维度，2 for 2D, 3 for 3D
        base_upsample: 原始上采样方法，'linear'（线性/三线性插值）或 'tc'（转置卷积）
        lambda_coeff: 感受野超参数λ，None则按论文设为1/特征图尺寸
    """

    def __init__(self, in_channels, out_channels, scale=2, dim=2, base_upsample='linear', lambda_coeff=None):
        super(DTC, self).__init__()
        self.scale = scale
        self.dim = dim
        self.base_upsample = base_upsample
        self.lambda_coeff = lambda_coeff  # 绑定实例属性
        self.g = dim  # 坐标生成轴数，2D=2,3D=3

        # 1. 卷积处理组件：统一原始上采样和DTC分支的通道数为out_channels
        self.conv_process = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False) if dim == 2 else \
            nn.Conv3d(in_channels, out_channels, 1, padding=0, bias=False)
        # 原始上采样分支的通道匹配卷积（核心修复：统一通道数）
        self.conv_base = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False) if dim == 2 else \
            nn.Conv3d(in_channels, out_channels, 1, padding=0, bias=False)

        # 2. 坐标生成组件：转置卷积生成offset(2g/3g)和weight(2g/3g)，论文中2D为4通道，3D为6通道
        self.tc_coord = nn.ConvTranspose2d(in_channels, 2 * self.g, kernel_size=scale, stride=scale,
                                           bias=False) if dim == 2 else \
            nn.ConvTranspose3d(in_channels, 2 * self.g, kernel_size=scale, stride=scale, bias=False)

        # 原始上采样的转置卷积（若base_upsample='tc'）
        self.base_tc = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=scale, stride=scale, bias=False) if (
                    dim == 2 and base_upsample == 'tc') else \
            (nn.ConvTranspose3d(in_channels, in_channels, kernel_size=scale, stride=scale, bias=False) if (
                        dim == 3 and base_upsample == 'tc') else None)

        # 初始化权重（论文默认初始化，保证训练稳定性）
        self._init_weights()

    def _init_weights(self):
        """权重初始化：卷积层xavier初始化，转置卷积层正态分布初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_base_upsample(self, x):
        """原始上采样：线性/三线性插值 或 转置卷积，保持通道数为in_channels"""
        if self.base_upsample == 'linear':
            return F.interpolate(x, scale_factor=self.scale, mode='bilinear' if self.dim == 2 else 'trilinear',
                                 align_corners=False)
        else:  # 'tc'
            return self.base_tc(x)

    def _generate_grid(self, x):
        """生成原始网格坐标Pn_Grid，范围[-1,1]（适配torch.nn.functional.grid_sample）"""
        if self.dim == 2:
            h, w = x.shape[2], x.shape[3]
            grid_h, grid_w = torch.meshgrid(torch.linspace(-1, 1, h * self.scale),
                                            torch.linspace(-1, 1, w * self.scale), indexing='ij')
            grid = torch.stack([grid_w, grid_h], dim=-1)  # [H*s, W*s, 2]
        else:
            d, h, w = x.shape[2], x.shape[3], x.shape[4]
            grid_d, grid_h, grid_w = torch.meshgrid(torch.linspace(-1, 1, d * self.scale),
                                                    torch.linspace(-1, 1, h * self.scale),
                                                    torch.linspace(-1, 1, w * self.scale), indexing='ij')
            grid = torch.stack([grid_w, grid_h, grid_d], dim=-1)  # [D*s, H*s, W*s, 3]
        return grid.unsqueeze(0).to(x.device)  # [1, D*s, H*s, W*s, 3/2]

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征图，2D形状[B, C, H, W]，3D形状[B, C, D, H, W]
        Returns:
            out: DTC融合后的上采样特征图，形状[B, out_C, H*s, W*s]/[B, out_C, D*s, H*s, W*s]
        """
        B, C = x.shape[0], x.shape[1]
        # 步骤1：原始上采样结果 + 通道匹配（核心修复：统一为out_channels）
        base_feat = self._get_base_upsample(x)  # [B, in_C, H*s, W*s] / [B, in_C, D*s, H*s, W*s]
        base_out = self.conv_base(base_feat)  # [B, out_C, H*s, W*s] / [B, out_C, D*s, H*s, W*s]

        # 步骤2：卷积处理组件提取特征 + 上采样
        conv_feat = self.conv_process(x)  # [B, out_C, H, W] / [B, out_C, D, H, W]
        conv_feat = F.interpolate(conv_feat, scale_factor=self.scale, mode='bilinear' if self.dim == 2 else 'trilinear',
                                  align_corners=False)  # 上采样到目标尺寸

        # 步骤3：坐标生成组件生成offset和weight
        coord_feat = self.tc_coord(x)  # [B, 2g, H*s, W*s]/[B, 2g, D*s, H*s, W*s]
        offset, weight = torch.chunk(coord_feat, 2, dim=1)  # 各为[B, g, ...]

        # 约束offset范围[-1,1]（tanh），weight范围[0,1]（sigmoid）
        offset = torch.tanh(offset)
        weight = torch.sigmoid(weight)

        # 调整offset/weight形状，适配grid_sample：[B, ..., g]
        if self.dim == 2:
            offset = offset.permute(0, 2, 3, 1)  # [B, H*s, W*s, 2]
            weight = weight.permute(0, 2, 3, 1)  # [B, H*s, W*s, 2]
        else:
            offset = offset.permute(0, 2, 3, 4, 1)  # [B, D*s, H*s, W*s, 3]
            weight = weight.permute(0, 2, 3, 4, 1)  # [B, D*s, H*s, W*s, 3]

        # 步骤4：计算动态坐标Pn_new = λ * offset * weight + Pn_Grid
        grid = self._generate_grid(x)  # [1, ..., g]
        if self.lambda_coeff is None:
            # 修复核心：将torch.Size转换为列表后计算均值，避免AttributeError
            feat_dims = x.shape[2:]  # 2D为(H,W)，3D为(D,H,W)
            feat_size = sum(feat_dims) / len(feat_dims)  # 计算尺寸均值（替代mean()）
            lambda_ = 1.0 / feat_size
        else:
            lambda_ = self.lambda_coeff
        new_grid = lambda_ * offset * weight + grid  # [B, ..., g]

        # 步骤5：grid_sample实现自适应上采样（核心修复：统一用bilinear，PyTorch自动适配3D为trilinear）
        # 兼容所有PyTorch版本，避免trilinear模式校验错误
        deform_feat = F.grid_sample(conv_feat, new_grid, mode='bilinear',
                                    padding_mode='zeros', align_corners=False)

        # 步骤6：融合原始上采样和DTC自适应上采样结果（论文核心设计）
        out = base_out + deform_feat  # 通道数均为out_channels，可正常相加
        return out


# ------------------------------
# DTC模块集成到U-Net解码器（示例）
# ------------------------------
class UNetDecoderBlock(nn.Module):
    """U-Net解码器块，替换原始上采样为DTC"""

    def __init__(self, in_channels, out_channels, scale=2, dim=2, base_upsample='linear'):
        super(UNetDecoderBlock, self).__init__()
        self.dtc = DTC(in_channels, out_channels, scale, dim, base_upsample)
        self.conv = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False) if dim == 2 else \
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if dim == 2 else nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dtc(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# ------------------------------
# 测试用例（可直接运行）
# ------------------------------
def test_dtc_2d():
    """测试2D DTC模块（适配ISIC/BUSI 2D医学图像分割）"""
    print("=" * 50)
    print("测试2D DTC模块（base_upsample=linear/tc）")
    # 模拟2D特征图：[B=2, C=64, H=32, W=32]，上采样尺度2
    x = torch.randn(2, 64, 32, 32)
    # 测试线性插值作为原始上采样
    dtc_2d_linear = DTC(in_channels=64, out_channels=32, scale=2, dim=2, base_upsample='linear')
    out_linear = dtc_2d_linear(x)
    # 测试转置卷积作为原始上采样
    dtc_2d_tc = DTC(in_channels=64, out_channels=32, scale=2, dim=2, base_upsample='tc')
    out_tc = dtc_2d_tc(x)
    # 验证输出形状：[2, 32, 64, 64]（H/W×2）
    assert out_linear.shape == (2, 32, 64, 64), f"2D DTC(linear)输出形状错误，预期(2,32,64,64)，实际{out_linear.shape}"
    assert out_tc.shape == (2, 32, 64, 64), f"2D DTC(tc)输出形状错误，预期(2,32,64,64)，实际{out_tc.shape}"
    print("2D DTC模块前向传播成功，输出形状：", out_linear.shape)
    print("=" * 50)


def test_dtc_3d():
    """测试3D DTC模块（适配BTCV15 3D医学图像分割）"""
    print("测试3D DTC模块（base_upsample=linear/tc）")
    # 模拟3D特征图：[B=1, C=128, D=16, H=32, W=32]，上采样尺度2
    x = torch.randn(1, 128, 16, 32, 32)
    # 测试三线性插值作为原始上采样
    dtc_3d_linear = DTC(in_channels=128, out_channels=64, scale=2, dim=3, base_upsample='linear')
    out_linear = dtc_3d_linear(x)
    # 测试3D转置卷积作为原始上采样
    dtc_3d_tc = DTC(in_channels=128, out_channels=64, scale=2, dim=3, base_upsample='tc')
    out_tc = dtc_3d_tc(x)
    # 验证输出形状：[1, 64, 32, 64, 64]（D/H/W×2）
    assert out_linear.shape == (
    1, 64, 32, 64, 64), f"3D DTC(linear)输出形状错误，预期(1,64,32,64,64)，实际{out_linear.shape}"
    assert out_tc.shape == (1, 64, 32, 64, 64), f"3D DTC(tc)输出形状错误，预期(1,64,32,64,64)，实际{out_tc.shape}"
    print("3D DTC模块前向传播成功，输出形状：", out_linear.shape)
    print("=" * 50)


def test_dtc_unet_decoder():
    """测试DTC集成到U-Net解码器块"""
    print("测试DTC集成到U-Net 2D/3D解码器块")
    # 2D解码器块测试
    decoder_2d = UNetDecoderBlock(in_channels=64, out_channels=32, scale=2, dim=2)
    x_2d = torch.randn(2, 64, 32, 32)
    out_2d = decoder_2d(x_2d)
    assert out_2d.shape == (2, 32, 64, 64), f"2D解码器输出形状错误，预期(2,32,64,64)，实际{out_2d.shape}"

    # 3D解码器块测试
    decoder_3d = UNetDecoderBlock(in_channels=128, out_channels=64, scale=2, dim=3)
    x_3d = torch.randn(1, 128, 16, 32, 32)
    out_3d = decoder_3d(x_3d)
    assert out_3d.shape == (1, 64, 32, 64, 64), f"3D解码器输出形状错误，预期(1,64,32,64,64)，实际{out_3d.shape}"

    print("U-Net 2D/3D解码器块（集成DTC）前向传播成功")
    print("2D解码器输出形状：", out_2d.shape)
    print("3D解码器输出形状：", out_3d.shape)
    print("=" * 50)


def test_dtc_lambda():
    """测试感受野超参数λ的有效性"""
    print("测试DTC感受野超参数λ")
    x = torch.randn(2, 64, 32, 32)
    # 自定义λ=0.5/1.0/2.0（论文推荐测试0.5/1/2/5）
    for lambda_ in [0.5, 1.0, 2.0]:
        dtc = DTC(in_channels=64, out_channels=32, scale=2, dim=2, lambda_coeff=lambda_)
        out = dtc(x)
        assert not torch.isnan(out).any(), f"λ={lambda_}时输出存在NaN，参数无效"
    print("超参数λ=0.5/1.0/2.0均有效，无NaN值")
    print("=" * 50)


# 运行所有测试用例
if __name__ == "__main__":
    test_dtc_2d()
    test_dtc_3d()
    test_dtc_unet_decoder()
    test_dtc_lambda()
    print("所有DTC模块测试用例执行成功！")