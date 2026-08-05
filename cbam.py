"""
CBAM: Convolutional Block Attention Module
论文: CBAM: Convolutional Block Attention Module (ECCV 2018)
用途: 在 YOLOv8 backbone 特征提取后添加通道+空间注意力，帮助模型聚焦火焰区域.

兼容环境: Python 3.12, PyTorch 2.10, ultralytics 8.3.18
"""

import torch
from torch import nn


class ChannelAttention(nn.Module):
    """通道注意力模块 (Channel Attention Module) 通过全局平均池化 + 最大池化 + 共享MLP，学习各通道的重要性权重.
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: 输入特征图的通道数
            reduction: MLP瓶颈压缩比，默认16.
        """
        super().__init__()
        mid = max(channels // reduction, 1)  # 防止 channels < reduction 时 mid=0

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 两个池化分支共享同一个 MLP
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(self.avg_pool(x))  # [B, C, 1, 1]
        max_out = self.shared_mlp(self.max_pool(x))  # [B, C, 1, 1]
        return self.sigmoid(avg_out + max_out)  # 通道注意力权重 [B, C, 1, 1]


class SpatialAttention(nn.Module):
    """空间注意力模块 (Spatial Attention Module) 通过沿通道维度做平均/最大池化后拼接，再卷积得到空间位置权重.
    """

    def __init__(self, kernel_size: int = 7):
        """
        Args:
            kernel_size: 空间卷积核大小，论文推荐7，也可用3.
        """
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size 必须为 3 或 7"
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        concat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        return self.sigmoid(self.conv(concat))  # 空间注意力权重 [B, 1, H, W]


class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module 先做通道注意力，再做空间注意力，均为残差乘法形式（不改变特征图尺寸和通道数）.

    在 yolov8m-cbam.yaml 中的使用方式:
        - [-1, 1, CBAM, [192]]   # channels=192，对应 P3 特征
        - [-1, 1, CBAM, [384]]   # channels=384，对应 P4 特征
        - [-1, 1, CBAM, [576]]   # channels=576，对应 SPPF 输出
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        """
        Args:
            channels: 输入/输出通道数（CBAM 不改变通道数）
            reduction: 通道注意力压缩比，默认16
            kernel_size: 空间注意力卷积核，默认7.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step1: 通道注意力（What to attend）
        x = x * self.channel_attention(x)
        # Step2: 空间注意力（Where to attend）
        x = x * self.spatial_attention(x)
        return x
