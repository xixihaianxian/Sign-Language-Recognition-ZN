import torch
from torch import nn
from torchvision import models
from typing import List
from loguru import logger

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
        self.conv1d_enhance=nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size,out_channels=hidden_size,kernel_size=3,stride=1,padding=int(i+1),groups=hidden_size,dilation=int(i+1)) for i in range(self.module_list_len)
        ])
        self.weights=nn.Parameter(torch.ones(size=(self.module_list_len,))/self.module_list_len,requires_grad=True)
        self.alpha=nn.Parameter(torch.zeros(size=(1,)),requires_grad=True)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x:torch.Tensor):
        r"""
        x: (batch_size,in_channels,temporal_length,height,width)
        """
        # 空间平均降维
        y=x.mean(-1).mean(-1)
        # 通道压缩
        y=self.conv1d_transform(y)
        # 多尺度时间卷积增强
        aggregated_out=0
        for module_index,module in enumerate(self.conv1d_enhance):
            aggregated_out+=module(y)*self.weights[module_index]
        y:torch.Tensor=self.conv1d_back(aggregated_out)
        return x*(torch.sigmoid(y.unsqueeze(-1).unsqueeze(-1))-0.5)*self.alpha
# 空间增强模块
class SSEM(nn.Module):
    def __init__(self,in_channels:int):
        super().__init__()
        if in_channels>=16:
            div_channels=in_channels//16
        else:
            logger.warning("in_channels needs to be greater than 16")
            div_channels=1
        self.conv3d_transform=nn.Conv3d(in_channels=in_channels,out_channels=div_channels,kernel_size=(1,1,1),bias=True)
        self.num_layers=3
        self.conv3d_enhance=nn.ModuleList([
            nn.Conv3d(in_channels=div_channels,out_channels=div_channels,kernel_size=(9,3,3),padding=(4,i+1,i+1),dilation=(1,i+1,i+1),groups=div_channels) for i in range(self.num_layers)
        ])
        self.weights=nn.Parameter(torch.ones(size=(self.num_layers,))/self.num_layers,requires_grad=True)
        self.conv3d_back=nn.Conv3d(in_channels=div_channels,out_channels=in_channels,kernel_size=(1,1,1))
        self.alpha=nn.Parameter(torch.ones(size=(1,)))
    def forward(self,x:torch.Tensor):
        # 通道压缩
        y=self.conv3d_transform(x)
        aggregated_out=torch.zeros_like(y)
        # 对每一个分支做卷积加权求和
        for module_index,module in enumerate(self.conv3d_enhance):
            aggregated_out+=module(y)*self.weights[module_index]
        y=self.conv3d_back(aggregated_out)
        gate=torch.sigmoid(y)-0.5
        y=x*gate*self.alpha
        return y
# 残差块
class BasicBlock(nn.Module):
    expansion=1
    def __init__(self,in_channels,middle_channels,stride=1,downsample=None):
        super().__init__()
        # 设置第一个三维卷积
        self.conv3d_1=conv3x3(in_channels=in_channels,out_channels=middle_channels,stride=stride)
        self.bn_1=nn.BatchNorm3d(middle_channels)
        # 设置第二个三维卷积
        self.conv3d_2=conv3x3(in_channels=middle_channels,out_channels=middle_channels)
        self.bn_2=nn.BatchNorm3d(middle_channels)
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample # 可选下采样模块
        # 增强模块
        self.tsem=TSEM(middle_channels)
        self.ssem=SSEM(middle_channels)
        # 默认下采样
        self.default_downsample=conv3x3(in_channels=in_channels,out_channels=middle_channels,stride=stride)
    def forward(self,x:torch.Tensor):
        residual = x
        y=self.conv3d_1(x)
        y=self.bn_1(y)
        y=self.relu(y)
        y=y+self.tsem(y)+self.ssem(y)
        y=self.conv3d_2(y)
        y=self.bn_2(y)
        if self.downsample is not None:
            residual=self.downsample(residual)
        else:
            logger.warning(f"Recommendation settings downsample!")
            residual=self.default_downsample(residual)
        y+=residual
        y=self.relu(y)
        return y
# 构建ResNet主网络
class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes=1000,in_channels=None):
        super().__init__()
        self.in_channels=in_channels if in_channels is not None else 64 # 初始化通道数
        self.conv3d_1=nn.Conv3d(in_channels=3,out_channels=64,kernel_size=(1,7,7),stride=(1,2,2),padding=(0,3,3),bias=False)
        self.bn_1=nn.BatchNorm3d(num_features=64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool3d(kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1))
    # 内部构造残差层
    def make_layers(self,block:BasicBlock,planes,blocks,stride=1):
        r"""
        用blocks个block来构造的残差模块
        """
        downsample=None
        if stride != 1 or self.in_channels!=planes*block.expansion:
            downsample=nn.Sequential(
                nn.Conv3d(in_channels=self.in_channels,out_channels=planes*block.expansion,kernel_size=(1,1,1),stride=(1,stride,stride),bias=False),
                nn.BatchNorm3d(planes*block.expansion)
            )
        layers=list()
        layers.append(block(in_channels=self.in_channels,middle_channels=planes,stride=stride,downsample=downsample))
        self.in_channels=planes*block.expansion
        for _ in range(1,blocks):
            layers.append(block(in_channels=self.in_channels,middle_channels=planes))
        layers=nn.Sequential(*layers)
        return layers
    def forward(self):
        pass
if __name__=="__main__":
    x=torch.randn(size=(1,128,50,64,64))
    # tsem=TSEM(input_channels=128)
    # ssem=SSEM(in_channels=128)
    # print(ssem(x).shape)
    basicblock=BasicBlock(in_channels=128,middle_channels=64)
    print(basicblock(x).shape)
