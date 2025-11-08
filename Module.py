import torch
from torch import nn
import torch.nn.functional as F
import copy
from loguru import logger
from typing import List
import math

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    'convnext_tiny': "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
    'convnext_base': "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
}

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x

# TemporalConv模块定义
class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size:int, convolution_type=2):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.convolution_type = convolution_type
        # 检测convolution_type是否符合要求
        if self.convolution_type<0 or self.convolution_type>3:
            logger.error(f"The convolution type exceeds the specified range")
            raise ValueError(f"The convolution type exceeds the specified range")
        # 不同的卷积结构类型
        if self.convolution_type == 0:
            self.kernel_size = ['K3']
        elif self.convolution_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.convolution_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]
        elif self.convolution_type == 3:
            self.kernel_size = ['K3', 'K3']
        # 构建modules
        modules = []
        for layer_index, operation_param in enumerate(self.kernel_size):
            # 当layer_index为0的时候，把in_channels设置为input_size
            in_channels=self.input_size if layer_index==0 else self.hidden_size
            # 获取操作方式
            operation=operation_param[0]
            # 获取操作的参数
            param=int(operation_param[1])
            # 构造模块
            if operation=="K":
            # operation=K表示添加卷积层
                modules.append(
                    nn.Conv1d(in_channels=in_channels,out_channels=self.hidden_size,kernel_size=param,stride=1,padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
            elif operation=="P":
            # operation=P表示添加池化层
                modules.append(
                    nn.MaxPool1d(kernel_size=param,ceil_mode=False)
                )
        self.temporal_convolution = nn.Sequential(*modules)
    # 计算出经过计算之后输出的长度
    def update_length(self, lengths:torch.Tensor)->List[torch.Tensor]:
        # 有效长度
        feature_length = copy.deepcopy(lengths)
        for operation_param in self.kernel_size:
            # 获取操作数
            operation=operation_param[0]
            # 获取操作参数
            param=int(operation_param[1])
            # 池化之后长度的变化
            if operation == 'P':
                feature_length = [torch.floor(length / 2).to(dtype=torch.int64) for length in feature_length]
            # 卷积之后长度的变化
            else:
                feature_length = [(length - param + 1) for length in feature_length]
        return feature_length
    def forward(self, frame_feature:torch.Tensor, lengths):
        visual_feature = self.temporal_convolution(frame_feature)
        lengths = self.update_length(lengths)
        return {
            "visual_feat": visual_feature,
            "feat_len": lengths,
        }

# 带权重归一化线性层
class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
    def forward(self, x):
        # 将权重规划划之后再相乘
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs

# 构造模块，和SEN里面的ResNet方法make_layer差不多
def make_layer(block:nn.Module,input_size:int,hidden_size:int,out_size:int,num_block:int,stride=1):
    downsample=None
    if stride!=1 or input_size!=out_size:
        downsample=nn.Sequential(
            nn.Conv3d(in_channels=input_size,out_channels=out_size,kernel_size=(1,1,1),stride=(1,stride,stride),bias=False),
            nn.BatchNorm3d(out_size)
        )
    layer=list()
    layer.append(block(input_size,hidden_size,out_size,stride,downsample))
    for _ in range(1,num_block):
        layer.append(block(out_size,hidden_size,out_size))
    # 将列表变为一个可以参加锻炼的module
    module=nn.Sequential(*layer)
    return module

# 设计一个不改变seq_len，height和width的3x3卷积
def conv3x3(in_channels,out_channels,stride=1):
    return nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(1,3,3),
        stride=(1,stride,stride),
        padding=(0,1,1),
        bias=False,
        dilation=1,
    )

# 3D ResNet 的基础残差块（BasicBlock）
class BasicBlock(nn.Module):
    r"""
    如果没有下采样操作的话，代码是会报错的，所以最好是加上下采样操作
    """
    expansion = 1
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super().__init__()
        self.conv3d_1=conv3x3(in_channels=in_channels,out_channels=out_channels,stride=stride)
        self.bn_1=nn.BatchNorm3d(out_channels)
        self.relu_1=nn.ReLU(inplace=True)
        self.conv3d_2=conv3x3(in_channels=out_channels,out_channels=out_channels*self.expansion)
        self.bn_2=nn.BatchNorm3d(out_channels*self.expansion)
        self.relu_2=nn.ReLU(inplace=True)
        self.downsample=downsample
    def forward(self,x):
        residual=x
        y=self.conv3d_1(x)
        y=self.bn_1(y)
        y=self.relu_1(y)
        y=self.conv3d_2(y)
        y=self.bn_2(y)
        if self.downsample is not None:
            residual=self.downsample(x)
        y+=residual
        y=self.relu_2(y)
        return y

# 3D特征增强模块
class GetCorrelation(nn.Module):
    def __init__(self,channels):
        super().__init__()
        reduction_channels=math.floor(channels//16)
        # 只改变通道数
        self.down_conv3d_1=nn.Conv3d(in_channels=channels,out_channels=reduction_channels,kernel_size=1,bias=False)
        self.down_conv3d_2=nn.Conv3d(in_channels=channels,out_channels=channels,kernel_size=1,bias=False)
        # 深度可分卷积，减少计算量
        self.spatial_aggregation_1=nn.Conv3d(in_channels=reduction_channels,out_channels=reduction_channels,kernel_size=(9,3,3),padding=(4,1,1),groups=reduction_channels)
    def forward(self):
        pass

if __name__=="__main__":
    # batch_size = 2
    # input_size = 64
    # hidden_size = 128
    # time_len = 50
    # x = torch.randn(batch_size, input_size, time_len)
    # lengths = torch.tensor([50,45])
    # model = TemporalConv(input_size, hidden_size, convolution_type=2)
    # output = model(x, lengths)
    # print(output)
    pass