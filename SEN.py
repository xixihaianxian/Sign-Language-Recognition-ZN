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
    def __init__(self,in_channels,middle_channels,stride=1,downsample=None,use_default_downsample=False):
        super().__init__()
        # 设置第一个三维卷积
        self.conv3d_1=conv3x3(in_channels=in_channels,out_channels=middle_channels,stride=stride)
        self.bn_1=nn.BatchNorm3d(middle_channels)
        # 设置第二个三维卷积
        self.conv3d_2=conv3x3(in_channels=middle_channels,out_channels=middle_channels*self.expansion)
        self.bn_2=nn.BatchNorm3d(middle_channels*self.expansion)
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample # 可选下采样模块
        # 增强模块
        self.tsem=TSEM(middle_channels)
        self.ssem=SSEM(middle_channels)
        # 默认下采样
        self.use_default_downsample=use_default_downsample
        self.default_downsample=nn.Sequential(
            conv3x3(in_channels=in_channels,out_channels=middle_channels,stride=stride),
            nn.BatchNorm3d(middle_channels)
        )
    def forward(self,x:torch.Tensor):
        residual = x
        y=self.conv3d_1(x)
        y=self.bn_1(y)
        y=self.relu(y)
        y=y+self.tsem(y)+self.ssem(y)
        y=self.conv3d_2(y)
        y=self.bn_2(y) # y此时的通道数是middle_channels*expansion
        # 没变换前的residual保持着原来的通道数
        if self.downsample is not None:
            # 变换之后residual的通道数应该和y保持相同
            residual=self.downsample(residual)
        elif self.downsample is None and self.use_default_downsample:
            logger.warning(f"Recommendation settings downsample!")
            residual=self.default_downsample(residual)
        y+=residual
        y=self.relu(y)
        # 测此时输出的y的通道数应该是middle_channels*expansion
        return y
# 构建ResNet主网络
class ResNet(nn.Module):
    r"""
    1.in_channels不要随便更改
    2.in_channels不是输入样本的通道数
    3.输入样本的通道数需要保持为3
    """
    def __init__(self,block,layers:List[int],num_classes:int=1000,in_channels=None):
        super().__init__()
        self.in_channels=in_channels if in_channels is not None else 64 # 初始化通道数
        self.conv3d_1=nn.Conv3d(in_channels=3,out_channels=64,kernel_size=(1,7,7),stride=(1,2,2),padding=(0,3,3),bias=False)
        self.bn_1=nn.BatchNorm3d(num_features=64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool3d(kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1))
        self.layer1=self.make_layer(block,planes=64,blocks=layers[0]) # 样本的height和width和seq_len都不发生改变，只有通道数发生了改变
        self.layer2=self.make_layer(block,planes=128,blocks=layers[1],stride=2)
        self.layer3=self.make_layer(block,planes=256,blocks=layers[2],stride=2)
        self.layer4=self.make_layer(block,planes=512,blocks=layers[3],stride=2)
        # 期望目标是（batch_size，channels，height，width）
        self.avgpool=nn.AvgPool2d(kernel_size=7,stride=1)
        self.fc=nn.Linear(in_features=512*block.expansion,out_features=num_classes)
        # 用于把张量展平
        self.flatten=nn.Flatten()
        # 对模块进行初始化
        for module in self.modules(): # 递归返回所有层包括自身
            if isinstance(module,nn.Conv3d) or isinstance(module,nn.Conv2d):
                nn.init.kaiming_normal_(module.weight,mode="fan_out",nonlinearity="relu")
            elif isinstance(module,nn.BatchNorm3d) or isinstance(module,nn.BatchNorm2d):
                nn.init.constant_(module.weight,1)
                nn.init.constant_(module.bias,0)
    # 内部构造残差层
    def make_layer(self,block:BasicBlock,planes,blocks,stride=1):
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
        # 经过一些列的计算之后，截止上一步，y的通道数是planes*expansion
        self.in_channels=planes*block.expansion # 此时把下一层的in_channels变为planes*expansion来对应上一层的y的通道数
        for _ in range(1,blocks):
            # 其实这里self.in_channels和之后输出的通道数数量是相同的
            layers.append(block(in_channels=self.in_channels,middle_channels=planes))
            # 输出的通道数仍然是planes*expansion，即使不添加downsample也不会有任何影响！！！
        layers=nn.Sequential(*layers)
        return layers
    def forward(self,x:torch.Tensor):
        r"""
        输入的x形状保证是(batch_size,channels,seq_len,height,width)
        """
        batch_size,channels,sequence_len,height,width=x.size()
        y=self.conv3d_1(x) # (batch_size,64,sequence_len,(height-1)2+1,(width-1)/2+1)
        y=self.bn_1(y)
        y=self.relu(y)
        y=self.maxpool(y) # (batch_size,64,sequence_len,(height-1)/2+1,(width-1)/2+1)
        y=self.layer1(y)
        y=self.layer2(y)
        y=self.layer3(y)
        y=self.layer4(y)
        y=y.transpose(1,2).contiguous()
        y=y.view((-1,)+y.size()[2:])
        y=self.avgpool(y)
        y=self.flatten(y)
        y=self.fc(y)
        return y
if __name__=="__main__":
    x=torch.randn(size=(5,3,20,224,224))
    resnet=ResNet(block=BasicBlock,layers=[1,2,3,4])
    print(resnet(x).shape)