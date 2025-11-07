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
        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]
        elif self.conv_type == 3:
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
    def update_length(self, lengths:List[int]):
        # 有效长度
        feature_length = copy.deepcopy(lengths)
        for operation_param in self.kernel_size:
            # 获取操作数
            operation=operation_param[0]
            # 获取操作参数
            param=int(operation_param[1])
            # 池化之后长度的变化
            if operation == 'P':
                feature_length = [math.floor(length / 2).int() for length in feature_length]
            # 卷积之后长度的变化
            else:
                feature_length = [math.floor(length - param + 1) for length in feature_length]
        return feature_length
    def forward(self, frame_feature:torch.Tensor, lengths):
        visual_feature = self.temporal_convolution(frame_feature)
        lengths = self.update_length(lengths)
        return {
            "visual_feat": visual_feature,
            "feat_len": lengths,
        }