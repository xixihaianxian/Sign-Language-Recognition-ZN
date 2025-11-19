import torch
from torch import nn
from torchvision import models
from BiLSTM import BiLSTMLayer
import Module
import SEN
import numpy as np
import Transformer
import os
from torch.hub import load_state_dict_from_url

class ModuleNet(nn.Module):
    def __init__(self, hidden_size, word_set_num:int, module_choice="Seq2Seq", device=torch.device("cuda"), data_set_name='RWTH', is_flag=False, download_weights=True, module_dir="~/model"):
        super().__init__()
        self.model_urls={
            "resnet34":"https://download.pytorch.org/models/resnet34-b627a593.pth", # weights=default
            "resnet18":"https://download.pytorch.org/models/resnet18-f37072fd.pth"
        }
        self.device = device # 设备
        self.module_choice = module_choice # 模型选择
        self.out_dim = word_set_num
        self.data_set_name = data_set_name
        self.log_soft_max = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.is_flag = is_flag
        self.probs_log = list()
        self.download_weights=download_weights # 是否提前下载模型，到指定位置
        self.module_dir=module_dir # 模型下载的位置
        # 如果需要提前下载好模型到指定位置
        if self.download_weights:
            os.path.exists(self.module_dir) or os.makedirs(self.module_dir)
            for name,url in self.model_urls.items():
                model_file=os.path.join(self.module_dir,f"{name}.pth")
                if os.path.exists(model_file):
                    pass
                else:
                    load_state_dict_from_url(url=url,model_dir=self.module_dir,file_name=f"{name}.pth")
        # 选择MSTNet，多尺度时空网络
        if "MSTNet" == self.module_choice:
            # 登录模型，使用resnet34作为特征提取网络
            self.feature_extraction = getattr(models, "resnet34")(weights=None)
            state_dict=torch.load(f=os.path.join(self.module_dir,f"resnet34.pth"))
            self.feature_extraction.load_state_dict(state_dict)
            self.feature_extraction.fc = Module.Identity()
            # 设置参数
            hidden_size = hidden_size
            input_size = hidden_size
            heads = 8
            semantic_layers = 2
            dropout = 0
            relative_position_encoding_k = 8
            # 第一组多尺度卷积
            self.conv1D1_1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
            self.conv1D1_2 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=5, stride=1, padding=2)
            self.conv1D1_3 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=7, stride=1, padding=3)
            self.conv1D1_4 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=9, stride=1, padding=4) # 数据的height，width均不改变
            # 二维卷积降维
            self.conv2D1 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(4, 2), stride=2,padding=0)
            # 第二组多尺度卷积
            self.conv1D2_1 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
            self.conv1D2_2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=5, stride=1, padding=2)
            self.conv1D2_3 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=7, stride=1, padding=3)
            self.conv1D2_4 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=9, stride=1, padding=4) # 数据的height，width均不改变
            # 二维卷积降维
            self.conv2D2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(4, 2), stride=2, padding=0)
            # 第一组的batchNorm，归一化
            self.batchNorm1d1_1 = nn.BatchNorm1d(hidden_size)
            self.batchNorm1d1_2 = nn.BatchNorm1d(hidden_size)
            self.batchNorm1d1_3 = nn.BatchNorm1d(hidden_size)
            self.batchNorm1d1_4 = nn.BatchNorm1d(hidden_size)
            # 第二组的batchNorm，归一化
            self.batchNorm1d2_1 = nn.BatchNorm1d(hidden_size)
            self.batchNorm1d2_2 = nn.BatchNorm1d(hidden_size)
            self.batchNorm1d2_3 = nn.BatchNorm1d(hidden_size)
            self.batchNorm1d2_4 = nn.BatchNorm1d(hidden_size)
            # 用于二维卷积的batchNorm，归一化
            self.batchNorm2d1 = nn.BatchNorm2d(hidden_size)
            self.batchNorm2d2 = nn.BatchNorm2d(hidden_size)
            # 使用relu函数作为非线性激活函数
            self.relu = nn.ReLU(inplace=True)
            # 时序模型建模
            self.temporal_model = Transformer.TransformerEncoder(hidden_size, heads, semantic_layers, dropout, relative_position_encoding_k)
            self.linear1 = nn.Linear(512, hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.batchNorm1d1 = nn.BatchNorm1d(hidden_size)
            self.batchNorm1d2 = nn.BatchNorm1d(hidden_size)
            self.classifier1 = Module.NormLinear(hidden_size, self.out_dim)
            self.classifier2 = Module.NormLinear(hidden_size, self.out_dim)
            if self.data_set_name == 'RWTH' or self.data_set_name == 'CE-CSL':
                self.classifier3 = Module.NormLinear(hidden_size, self.out_dim)
                self.classifier4 = Module.NormLinear(input_size, self.out_dim)
        elif "VAC" == self.module_choice:
            # 登录模型
            self.conv2d = getattr(models, "resnet18")(weights=None)
            state_dict=torch.load(f=os.path.join(self.module_dir,"resnet18.pth"))
            self.load_state_dict(state_dict)
            # 设置参数
            hidden_size = hidden_size
            self.conv2d.fc = Module.Identity()
            self.conv1d = Module.TemporalConv(input_size=512, hidden_size=hidden_size, convolution_type=2)
            self.temporal_model = BiLSTMLayer(rnn_type='LSTM', embedding_size=hidden_size, hidden_size=hidden_size, num_layers=2, bidirectional=True)
            self.classifier = Module.NormLinear(hidden_size, self.out_dim)
            self.classifier1 = self.classifier
        elif "CorrNet" == self.module_choice:
            hidden_size = hidden_size
            self.conv2d = Module.resnet18corr()
            self.conv2d.fc = Module.Identity()
            self.conv1d = Module.TemporalConv(input_size=512, hidden_size=hidden_size, convolution_type=2)
            self.temporal_model = BiLSTMLayer(rnn_type='LSTM', embedding_size=hidden_size, hidden_size=hidden_size, num_layers=2, bidirectional=True)
            self.classifier = Module.NormLinear(hidden_size, self.out_dim)
            self.classifier1 = self.classifier
        elif "MAM-FSD" == self.module_choice:
            hidden_size = hidden_size
            self.conv2d = Module.resnet34mam()
            self.conv2d.fc = Module.Identity()
            self.conv1d = Module.TemporalConv(input_size=512, hidden_size=hidden_size, convolution_type=2)
            self.temporal_model = BiLSTMLayer(rnn_type='LSTM', embedding_size=hidden_size, hidden_size=hidden_size, num_layers=2, bidirectional=True)
            self.classifier = Module.NormLinear(hidden_size, self.out_dim)
            self.classifier1 = self.classifier
            self.conv1 = nn.Conv3d(64, 128, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0), bias=False)
            self.conv2 = nn.Conv3d(128, 256, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0), bias=False)
            self.conv3 = nn.Conv3d(256, 512, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0), bias=False)
            self.batchNorm3d1 = nn.BatchNorm3d(128)
            self.batchNorm3d2 = nn.BatchNorm3d(256)
            self.batchNorm3d3 = nn.BatchNorm3d(512)
            self.reLU = nn.ReLU(inplace=True)
        elif "SEN" == self.module_choice:
            hidden_size = hidden_size
            self.conv2d = SEN.resnet18()
            self.conv2d.fc = Module.Identity()
            self.conv1d = Module.TemporalConv(input_size=512, hidden_size=hidden_size, convolution_type=2)
            self.temporal_model = BiLSTMLayer(rnn_type='LSTM', embedding_size=hidden_size, hidden_size=hidden_size, num_layers=2, bidirectional=True)
            self.classifier = nn.Linear(in_features=hidden_size, out_features=self.out_dim)
            self.classifier1 = nn.Linear(in_features=hidden_size, out_features=self.out_dim)
        elif "TFNet" == self.module_choice:
            hidden_size = hidden_size
            self.conv2d = Module.resnet34mam()
            self.conv2d.fc = Module.Identity()
            self.conv1d = Module.TemporalConv(input_size=512, hidden_size=hidden_size, convolution_type=2)
            self.conv1d1 = Module.TemporalConv(input_size=512, hidden_size=hidden_size, convolution_type=2)
            self.temporal_model = BiLSTMLayer(rnn_type='LSTM', embedding_size=hidden_size, hidden_size=hidden_size, num_layers=2, bidirectional=True)
            self.temporal_model1 = BiLSTMLayer(rnn_type='LSTM', embedding_size=hidden_size, hidden_size=hidden_size, num_layers=2, bidirectional=True)
            self.classifier11 = Module.NormLinear(hidden_size, self.out_dim)
            self.classifier22 = self.classifier11
            self.classifier33 = Module.NormLinear(hidden_size, self.out_dim)
            self.classifier44 = self.classifier33
            self.classifier55 = Module.NormLinear(hidden_size, self.out_dim)
            self.reLU = nn.ReLU(inplace=True)
    # 填充
    def pad(self, tensor:torch.Tensor, length:int)->torch.Tensor:
        number=tensor.size(0)
        if number<length:
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])
        # 当length大于number时对数据进行裁剪（保留目前不知道是否可以这样操作）
        else:
            return tensor[:length]
    def forward(self, seq_data, data_len=None, is_train=True):
        out_data_1 = None
        out_data_2 = None
        out_data_3 = None
        log_probs_1 = None
        log_probs_2 = None
        log_probs_3 = None
        log_probs_4 = None
        log_probs_5 = None
        # 设置len_x
        len_x = data_len
        # 获取batch_size,temp,channels,height,width
        batch_size, temp, channels, height, width = seq_data.shape
        # 选择MSTNet模型，注意这不是一个经典的MSTNet，可以理解为Multi-Stream Temporal Network（动作识别）
        if "MSTNet" == self.module_choice:
            inputs = seq_data.reshape(batch_size * temp, channels, height, width)
            # 划分数据集
            x = torch.cat([inputs[len_x[0] * index: len_x[0] * index + length] for index, length in enumerate(len_x)])
            n = len(x)
            indices = np.arange(n)
            np.random.shuffle(indices)
            train_index = indices[: int(n * 0.5)]
            train_index = sorted(train_index)
            test_index = indices[int(n * 0.5):]
            test_index = sorted(test_index)
            train_data = x[train_index, :, :, :] # shape(batch_size*temp, channels, height, width)
            test_data = x[test_index, :, :, :]
            # 训练集特征提取
            train_data = self.feature_extraction(train_data)
            # 测试集特征提取
            with torch.no_grad():
                test_data = self.feature_extraction(test_data)
            shape = train_data.shape
            # x1转移到cuda上
            x1 = torch.zeros(size=((shape[0] // 1) * 2, shape[1])).to(device=self.device) # shape(batch_size*temp,feature_dim)
            for i in range(len(train_index)):
                x1[train_index[i], :] = train_data[i, :]
            for i in range(len(test_index)):
                x1[test_index[i], :] = test_data[i, :]
            # 计算之后x1上面包含了所有的经过特征提取之后的train_data和test_data，同时排列的顺序也是按照x的顺序来排列
            # 序列长度标准化，保持在len_x[0]
            frame_wise = torch.cat([self.pad(x1[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0]) for idx, length in enumerate(len_x)])
            # 修改frame_wise的形状
            frame_wise = frame_wise.reshape(batch_size, temp, -1)
            # 1
            frame_wise = self.linear1(frame_wise).transpose(1, 2) # 512->hidden_size
            frame_wise = self.batchNorm1d1(frame_wise)
            frame_wise = self.relu(frame_wise).transpose(1, 2)
            frame_wise = self.linear2(frame_wise).transpose(1, 2) # hidden_size->hidden_size
            frame_wise = self.batchNorm1d2(frame_wise)
            frame_wise = self.relu(frame_wise) # (batch_size, temp, hidden_size)
            input_data = self.conv1D1_1(frame_wise) # (batch_size, temp, hidden_size)
            input_data = self.batchNorm1d1_1(input_data)
            input_data = self.relu(input_data)
            gloss_candidate = input_data.unsqueeze(2) # (batch_size, temp, 1, hidden_size)
            input_data = self.conv1D1_2(frame_wise) # (batch_size, temp, 1, hidden_size)
            input_data = self.batchNorm1d1_2(input_data)
            input_data = self.relu(input_data)
            tmp_data = input_data.unsqueeze(2) # (batch_size, temp, 1, 1, hidden_size)
            gloss_candidate = torch.cat([gloss_candidate, tmp_data], dim=2)
            input_data = self.conv1D1_3(frame_wise)
            input_data = self.batchNorm1d1_3(input_data)
            input_data = self.relu(input_data)
            tmp_data = input_data.unsqueeze(2)
            gloss_candidate = torch.cat([gloss_candidate, tmp_data], dim=2)
            input_data = self.conv1D1_4(frame_wise)
            input_data = self.batchNorm1d1_4(input_data)
            input_data = self.relu(input_data)
            tmp_data = input_data.unsqueeze(2)
            gloss_candidate = torch.cat([gloss_candidate, tmp_data], dim=2)
            input_data = self.conv2D1(gloss_candidate)
            input_data = self.batchNorm2d1(input_data)
            input_data_1 = self.relu(input_data).squeeze(2)
            # 2
            input_data = self.conv1D2_1(input_data_1)
            input_data = self.batchNorm1d2_1(input_data)
            input_data = self.relu(input_data)
            gloss_candidate = input_data.unsqueeze(2)
            input_data = self.conv1D2_2(input_data_1)
            input_data = self.batchNorm1d2_2(input_data)
            input_data = self.relu(input_data)
            tmp_data = input_data.unsqueeze(2)
            gloss_candidate = torch.cat([gloss_candidate, tmp_data], dim=2)
            input_data = self.conv1D2_3(input_data_1)
            input_data = self.batchNorm1d2_3(input_data)
            input_data = self.relu(input_data)
            tmp_data = input_data.unsqueeze(2)
            gloss_candidate = torch.cat([gloss_candidate, tmp_data], dim=2)
            input_data = self.conv1D2_4(input_data_1)
            input_data = self.batchNorm1d2_4(input_data)
            input_data = self.relu(input_data)
            tmp_data = input_data.unsqueeze(2)
            gloss_candidate = torch.cat([gloss_candidate, tmp_data], dim=2)
            input_data = self.conv2D2(gloss_candidate)
            input_data = self.batchNorm2d2(input_data)
            input_data = self.relu(input_data).squeeze(2)
            # if not self.data_set_name == 'CSL':  这里可能存在错误
            if "CSL" in self.data_set_name:
                length = torch.cat(len_x, dim=0) // 4
                x = input_data.permute(0, 2, 1)
            else:
                length = (torch.cat(len_x, dim=0) // 4) - 6
                x = input_data.permute(0, 2, 1)
                x = x[:, 3:-3, :]
            outputs = self.temporal_model(x)
            outputs = outputs.permute(1, 0, 2)
            log_probs_1 = self.classifier1(outputs)
            outputs = x.permute(1, 0, 2)
            log_probs_2 = self.classifier2(outputs)
            # if not self.data_set_name == 'CSL': 这里可能存在错误
            if "CSL" in self.data_set_name:
                outputs = input_data_1.permute(2, 0, 1)
                log_probs_3 = self.classifier3(outputs)
                outputs = frame_wise.permute(2, 0, 1)
                log_probs_4 = self.classifier4(outputs)
            log_probs_5 = log_probs_1
        # 选择模型VAC
        elif "VAC" == self.module_choice:
            inputs = seq_data.reshape(batch_size * temp, channels, height, width)
            x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + length] for idx, length in enumerate(len_x)])
            x = self.conv2d(x)
            frame_wise = torch.cat([self.pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0]) for idx, length in enumerate(len_x)])
            frame_wise = frame_wise.reshape(batch_size, temp, -1).transpose(1, 2)
            conv1d_outputs = self.conv1d(frame_wise, len_x)
            # x: T, B, C
            x = conv1d_outputs['visual_feat']
            length = conv1d_outputs['feat_len']
            x = x.permute(2, 0, 1)
            length = torch.cat(length, dim=0)
            outputs = self.temporal_model(x, length)
            encoder_prediction = self.classifier(outputs['predictions'])
            log_probs_1 = encoder_prediction
            encoder_prediction = self.classifier1(x)
            log_probs_2 = encoder_prediction
        elif "CorrNet" == self.module_choice:
            x = seq_data.transpose(1, 2)
            frame_wise = self.conv2d(x)
            frame_wise = frame_wise.reshape(batch_size, temp, -1).transpose(1, 2)
            conv1d_outputs = self.conv1d(frame_wise, len_x)
            # x: T, B, C
            x = conv1d_outputs['visual_feat']
            length = conv1d_outputs['feat_len']
            x = x.permute(2, 0, 1)
            length = torch.cat(length, dim=0)
            outputs = self.temporal_model(x, length)
            encoder_prediction = self.classifier(outputs['predictions'])
            log_probs_1 = encoder_prediction
            encoder_prediction = self.classifier1(x)
            log_probs_2 = encoder_prediction
        elif "MAM-FSD" == self.module_choice:
            x = seq_data.transpose(1, 2)
            frame_wise, out_data_1, out_data_2, out_data_3 = self.conv2d(x)
            tmpOut = self.conv1(out_data_1[0])
            tmpOut = self.batchNorm3d1(tmpOut)
            out_data_1[0] = self.reLU(tmpOut)
            tmpOut = self.conv2(out_data_2[0])
            tmpOut = self.batchNorm3d2(tmpOut)
            out_data_2[0] = self.reLU(tmpOut)
            tmpOut = self.conv3(out_data_3[0])
            tmpOut = self.batchNorm3d3(tmpOut)
            out_data_3[0] = self.reLU(tmpOut)
            frame_wise = frame_wise.reshape(batch_size, temp, -1).transpose(1, 2)
            conv1d_outputs = self.conv1d(frame_wise, len_x)
            # x: T, B, C
            x = conv1d_outputs['visual_feat']
            length = conv1d_outputs['feat_len']
            x = x.permute(2, 0, 1)
            length = torch.cat(length, dim=0)
            outputs = self.temporal_model(x, length)
            encoder_prediction = self.classifier(outputs['predictions'])
            log_probs_1 = encoder_prediction
            encoder_prediction = self.classifier1(x)
            log_probs_2 = encoder_prediction
            log_probs_3 = None
            log_probs_4 = None
        elif "SEN" == self.module_choice:
            x = seq_data.transpose(1, 2)
            frame_wise = self.conv2d(x)
            frame_wise = frame_wise.reshape(batch_size, temp, -1).transpose(1, 2)
            conv1d_outputs = self.conv1d(frame_wise, len_x)
            # x: T, B, C
            x = conv1d_outputs['visual_feat']
            length = conv1d_outputs['feat_len']
            x = x.permute(2, 0, 1)
            length = torch.cat(length, dim=0)
            outputs = self.temporal_model(x, length)
            encoder_prediction = self.classifier(outputs['predictions'])
            log_probs_1 = encoder_prediction
            encoder_prediction = self.classifier1(x)
            log_probs_2 = encoder_prediction
        elif "TFNet" == self.module_choice:
            x = seq_data.transpose(1, 2)
            frame_wise, out_data_1, out_data_2, out_data_3 = self.conv2d(x)
            frame_wise = frame_wise.reshape(batch_size, temp, -1).transpose(1, 2)
            # 傅里叶变换
            framewise1 = frame_wise.transpose(1, 2).float()
            X = torch.fft.fft(framewise1, dim=-1, norm="forward")
            X = torch.abs(X)
            framewise1 = X.transpose(1, 2)
            conv1d_outputs = self.conv1d(frame_wise, len_x)
            # x: T, B, C
            x = conv1d_outputs['visual_feat']
            length = conv1d_outputs['feat_len']
            x = x.permute(2, 0, 1)
            length = torch.cat(length, dim=0)
            conv1d_outputs1 = self.conv1d1(framewise1, len_x)
            # x: T, B, C
            x1 = conv1d_outputs1['visual_feat']
            x1 = x1.permute(2, 0, 1)
            outputs = self.temporal_model(x, length)
            outputs1 = self.temporal_model1(x1, length)
            encoder_prediction = self.classifier11(outputs['predictions'])
            log_probs_1 = encoder_prediction
            encoder_prediction = self.classifier22(x)
            log_probs_2 = encoder_prediction
            encoder_prediction = self.classifier33(outputs1['predictions'])
            log_probs_3 = encoder_prediction
            encoder_prediction = self.classifier44(x1)
            log_probs_4 = encoder_prediction
            x2 = outputs['predictions'] + outputs1['predictions']
            log_probs_5 = self.classifier55(x2)
            if not is_train:
                log_probs_1 = log_probs_5
        return log_probs_1, log_probs_2, log_probs_3, log_probs_4, log_probs_5, length, out_data_1, out_data_2, out_data_3

if __name__=="__main__":
    x=torch.randn(size=(20,3,512,512))
    resnet34=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet34.fc=Module.Identity()
    print(resnet34)