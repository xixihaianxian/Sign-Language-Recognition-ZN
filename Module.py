import torch
from torch import nn
import torch.nn.functional as F
import copy
from typing import List
import math
from loguru import logger
from torch.hub import load_state_dict_from_url

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
        # 判断channels是否合理
        if 0<=channels<16:
            logger.warning(f"channels的大小似乎不太合理，请检查channels的大小!")
        elif channels<0:
            logger.error(f"channels不能为负数！")
            raise ValueError(f"channels不能为负数！")
        # 计算中间层的通道数
        reduction_channels=math.floor(channels//16)
        # 只改变通道数
        self.down_conv3d_1=nn.Conv3d(in_channels=channels,out_channels=reduction_channels,kernel_size=1,bias=False)
        self.down_conv3d_2=nn.Conv3d(in_channels=channels,out_channels=channels,kernel_size=1,bias=False)
        # 深度可分卷积，减少计算量
        self.spatial_aggregation_1=nn.Conv3d(in_channels=reduction_channels,out_channels=reduction_channels,kernel_size=(9,3,3),padding=(4,1,1),groups=reduction_channels)
        self.spatial_aggregation_2 = nn.Conv3d(in_channels=reduction_channels, out_channels=reduction_channels, kernel_size=(9, 3, 3), padding=(4, 2, 2), dilation=(1, 2, 2), groups=reduction_channels)
        self.spatial_aggregation_3 = nn.Conv3d(in_channels=reduction_channels, out_channels=reduction_channels, kernel_size=(9, 3, 3), padding=(4, 3, 3), dilation=(1, 3, 3), groups=reduction_channels)
        # 定义权重
        self.weight_1=nn.Parameter(torch.ones(size=(3,))/3,requires_grad=True)
        self.weight_2=nn.Parameter(torch.ones(size=(2,))/2,requires_grad=True)
        # 使用3维卷积将通道数还原
        self.conv3d_back=nn.Conv3d(in_channels=reduction_channels,out_channels=channels,kernel_size=1,bias=False)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x:torch.Tensor):
        x2=self.down_conv3d_2(x)
        affinities_1=torch.einsum("bcthw,bctsd->bthwsd",x,torch.concat((x2[:,:,1:],x2[:,:,-1:]),dim=2)) # 重复最后一帧
        affinities_2=torch.einsum("bcthw,bctsd->bthwsd",x,torch.concat((x2[:,:,:1],x2[:,:,:-1]),dim=2)) # 重复第一帧
        term1=torch.einsum('bctsd,bthwsd->bcthw',
                           torch.concat([x2[:, :, 1:], x2[:, :, -1:]], 2),
                           self.sigmoid(affinities_1) - 0.5) * self.weight_2[0]
        term2=torch.einsum("bctsd,bthwsd->bcthw",
                           torch.concat([x2[:, :, :1], x2[:, :, :-1]], 2),
                           self.sigmoid(affinities_2) - 0.5)*self.weight_2[1]
        feature=term1+term2
        y=self.down_conv3d_1(x)
        aggregated_y=self.spatial_aggregation_1(y) * self.weight_1[0] + self.spatial_aggregation_2(y) * self.weight_1[1] + self.spatial_aggregation_3(y) * self.weight_1[2]
        aggregated_y=self.conv3d_back(aggregated_y)
        return feature*(self.sigmoid(aggregated_y)-0.5)

# 构建ResNet模块（普通的ResNet）
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        self.conv3d_1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.bn_1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer_1 = self.make_layer(block, 64, layers[0])
        self.layer_2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer_3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer_4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm3d) or isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    def make_layer(self, block, planes, blocks, stride=1):
        downsample=None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = list()
        layers.append(block(self.in_channels, planes, stride, downsample))
        self.in_channels = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        y = self.conv3d_1(x)
        y = self.bn_1(y)
        y = self.relu(y)
        y = self.maxpool(y)
        y = self.layer_1(y)
        y = self.layer_2(y)
        y = self.layer_3(y)
        y = self.layer_4(y)
        y = y.transpose(1, 2).contiguous()
        y = y.view((-1,) + y.size()[2:])
        y = self.avgpool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y

# 融合帧间相关性的3D ResNet
class ResNetCorr(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        self.conv3d_1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.bn_1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        # 残差块堆叠make_layer
        self.layer_1 = self.make_layer(block, 64, layers[0])
        self.layer_2 = self.make_layer(block, 128, layers[1], stride=2)
        # 计算时间相关性
        self.corr_1 = GetCorrelation(self.in_channels)
        self.layer_3 = self.make_layer(block, 256, layers[2], stride=2)
        self.corr_2 = GetCorrelation(self.in_channels)
        self.alpha = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.layer_4 = self.make_layer(block, 512, layers[3], stride=2)
        self.corr_3 = GetCorrelation(self.in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = list()
        layers.append(block(self.in_channels, planes, stride, downsample))
        self.in_channels = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        y = self.conv3d_1(x)
        y = self. bn_1(y)
        y = self.relu(y)
        y = self.maxpool(y)
        y = self.layer_1(y)
        y = self.layer_2(y)
        y = y + self.corr_1(y) * self.alpha[0]
        y = self.layer_3(y)
        y = y + self.corr_2(y) * self.alpha[1]
        y = self.layer_4(y)
        y = y + self.corr_3(y) * self.alpha[2]
        y = y.transpose(dim0=1, dim1=2).contiguous()
        y = y.view((-1,) + y.size()[2:])
        y = self.avgpool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y

# 在ResNet34的基础上加上多级运动的注意力机制
class ResNet34MAM(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet34MAM, self).__init__()
        self.in_channels = 64
        self.motorAttention_1 = MotorAttention(3, 16)
        self.conv3d_1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),bias=False)
        self.bn_1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.layer_1 = self.make_layer(block, 64, layers[0])
        self.motorAttention_2 = MotorAttention(64, 64)
        self.layer_2 = self.make_layer(block, 128, layers[1], stride=2)
        self.motorAttention_3 = MotorAttention(128, 64)
        self.layer_3 = self.make_layer(block, 256, layers[2], stride=2)
        self.motorAttention_4 = MotorAttention(256, 64)
        self.layer_4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm3d) or isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = list()
        layers.append(block(self.in_channels, planes, stride, downsample))
        self.in_channels = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels,planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        out_data_1 = list()
        out_data_2 = list()
        out_data_3 = list()
        out = self.motorAttention_1(x)
        out = self.conv3d_1(out)
        out = self.bn_1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer_1(out)
        out = self.motorAttention_2(out)
        out_data_1.append(out)
        out = self.layer_2(out)
        out = self.motorAttention_3(out)
        out_data_1.append(out)
        out_data_2.append(out)
        out = self.layer_3(out)
        out = self.motorAttention_4(out)
        out_data_2.append(out)
        out_data_3.append(out)
        out = self.layer_4(out)
        out_data_3.append(out)
        out = out.transpose(1, 2).contiguous()
        out = out.view((-1,) + out.size()[2:])
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, out_data_1, out_data_2, out_data_3

# 针对时间维度的注意力模块，增强输入增量在时间维度度上的重要性
class MotorAttention(nn.Module):
    def __init__(self,in_channels,hidden_channels):
        super().__init__()
        kernel=3
        padding=1
        self.conv3d_1=nn.Conv3d(in_channels=in_channels,out_channels=hidden_channels,kernel_size=(kernel,1,1),stride=1,padding=(padding,0,0))
        self.conv3d_2=nn.Conv3d(in_channels=hidden_channels,out_channels=hidden_channels,kernel_size=(kernel,1,1),stride=1,padding=(padding,0,0))
        self.conv3d_3=nn.Conv3d(in_channels=hidden_channels,out_channels=hidden_channels,kernel_size=(kernel,1,1),stride=1,padding=(padding,0,0))
        self.conv3d_4=nn.Conv3d(in_channels=hidden_channels,out_channels=in_channels,kernel_size=(kernel,1,1),stride=1,padding=(padding,0,0))
        self.relu=nn.LeakyReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x:torch.Tensor):
        y=self.conv3d_1(x)
        y=self.relu(y)
        y=self.conv3d_2(y)
        y=self.relu(y)
        y=self.conv3d_3(y)
        y=self.relu(y)
        y=self.conv3d_4(y)
        y=self.sigmoid(y)
        out=x*y
        return out

# 定义修饰器
def resnet_loader(function):
    r"""
    用于自动加载resnet18的权重
    """
    def wrapper(*args,**kwargs):
        custom_module:nn.Module=function(*args,**kwargs)
        # 确定函数的名称
        function_name=function.__name__
        # 确定需要登录的模型名称
        if function_name=="resnet34mam":
            model_name="resnet34"
        else:
            model_name="resnet18"
        module_url=model_urls.get(model_name)
        # 判断resnet18或者resnet34是否存在
        if module_url is None:
            logger.error(f"{model_name} does not exist, please check the model urls!")
            raise KeyError(f"{model_name} does not exist, please check the model urls!")
        # 登录预训练模型参数
        state_dict=load_state_dict_from_url(url=module_url,model_dir="resnet",file_name=f"{model_name}.pth")
        for name,param in state_dict.items():
            if "conv" in name or "downsample.0.weight" in name:
                state_dict[name]=param.unsqueeze(2)
        if function_name=="resnet34mam":
            # 获取模型默认的参数
            default_state_dict=custom_module.state_dict()
            pretrained_dict={name:param for name,param in state_dict.items() if name in default_state_dict.keys()}
            # 更新默认的参数
            default_state_dict.update(pretrained_dict)
            # 将更新之后的参数赋值给state_dict
            state_dict=default_state_dict
        # 登录预训练参数
        custom_module.load_state_dict(state_dict,strict=False)
        return custom_module
    return wrapper

# ResNet登录默认的resnet18参数
@ resnet_loader
def resnet18(**kwargs):
    custom_module=ResNet(block=BasicBlock,layers=[2,2,2,2],**kwargs)
    return custom_module

# ResNetCorr登录默认的resnet18参数
@ resnet_loader
def resnet18corr(**kwargs):
    custom_module=ResNetCorr(block=BasicBlock,layers=[2,2,2,2],**kwargs)
    return custom_module

# ResNet34MAM登录默认的resnet34参数
@ resnet_loader
def resnet34mam(**kwargs):
    custom_module=ResNet34MAM(block=BasicBlock,layers=[3, 4, 6, 3],**kwargs)
    return custom_module

if __name__=="__main__":
    batch_size = 2
    channels = 3
    frames = 16
    height = 224
    width = 224
    x = torch.randn(batch_size, channels, frames, height, width)
    model = ResNet34MAM(block=BasicBlock,layers=[2,2,2,2])
    output = model(x)
    print(output[1])