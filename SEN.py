import torch
from torch import nn
from torchvision import models
from typing import List

# 只在空间（H,W）做 3×3 卷积，时间维度（T）保持不变
def conv3x3(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(1,3,3),
        stride=(1,stride,stride),
        padding=(0,1,1),
        bias=bias
    )

# 时间增强模块
class TSEM(nn.Module):
    r"""
    input_size: 输入特征通道数
    hidden_size: 内部瓶颈通道
    kernel_size: 多尺度 1D 卷积核大小列表，用于不同时间感受野
    alpha: 控制增强维度
    """
    def __init__(self,input_channels:int):
        super().__init__()
        hidden_size=input_channels//16 # 计算隐藏通道数
        self.conv1d_transform=nn.Conv1d(in_channels=input_channels,out_channels=hidden_size,kernel_size=1,stride=1,padding=0)
        self.conv1d_back=nn.Conv1d(in_channels=hidden_size,out_channels=input_channels,kernel_size=1,stride=1,padding=0)
        self.module_list_len=5
        nn.ModuleList()
    def forward(self):
        pass

# 空间增强模块
class SSEM(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        pass

# 残差块
class BasicBlock(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        pass

# 构建ResNet主网络
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        pass