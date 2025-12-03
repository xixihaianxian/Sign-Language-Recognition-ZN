import torch
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from typing import Dict,Any
import readconfig
import DataPreprocessing
import VideoEnhancement
import Net
from torch import nn
from torch import optim
import math
from loguru import logger
import decode

# 设置随机种子，提高代码的复现可能性
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False # 禁止cuDNN自动优化功能
    torch.backends.cudnn.deterministic = True # 强制cuDNN使用确定性算法

def stable(dataloader,seed):
    seed_torch(seed)
    return dataloader

# 损失函数
def loss_function(module_choice:str):
    pad_idx=0
    ctc_loss=None
    kld=None
    mean_loss=None
    if module_choice=="MSTNet":
        ctc_loss=nn.CTCLoss(blank=pad_idx,reduction="mean",zero_infinity=True)
    elif module_choice=="VAC" or module_choice=="CorrNet" or module_choice=="MAM-FSD" or module_choice=="SEN" or module_choice=="TFNet":
        ctc_loss=nn.CTCLoss(blank=pad_idx,reduction="none",zero_infinity=True)
        kld=DataPreprocessing.SeqKD(T=8)
        if module_choice=="MAM-FSD":
            mean_loss=nn.MSELoss(reduction="mean")
    return ctc_loss,kld,mean_loss

# 构造优化器
def optimizer_function(module:torch.nn.Module,learning_rate:float,weight_decay:float)->optim.optimizer.Optimizer:
    return optim.Adam(module.parameters(),lr=learning_rate,weight_decay=weight_decay)

# 训练
def train(config_params:Dict[str,Any],is_train=True):
    # 初始化数据路径参数
    train_data_path=config_params["trainDataPath"]
    valid_data_path=config_params["valid_data_path"]
    test_data_path=config_params["test_data_path"]
    # 初始化标签路径参数
    train_label_path=config_params["train_label_path"]
    valid_label_path=config_params["valid_label_path"]
    test_label_path=config_params["test_label_path"]
    # 初始化模型位置参数
    best_module_path = config_params["bestModuleSavePath"]
    current_module_path = config_params["currentModuleSavePath"]
    # 初始化参数
    device:torch.device = config_params["device"]
    hidden_size = int(config_params["hiddenSize"])
    lr = float(config_params["lr"])
    batch_size = int(config_params["batchSize"])
    num_workers = int(config_params["numWorkers"])
    pin_memory = bool(int(config_params["pinMemory"]))
    module_choice = config_params["moduleChoice"]
    data_set_name = config_params["dataSetName"]
    max_num_states = 1
    # RWTH数据集的处理
    if data_set_name == "RWTH":
        source_file_path = './evaluation/wer/evalute'
        if is_train:
            file_name = f"output-hypothesis-dev.ctm"
        else:
            file_name = f"output-hypothesis-test.ctm"
        file_path = os.path.join(source_file_path, file_name)
    # 对RWTH-T数据集的处理
    elif data_set_name == "RWTH-T":
        source_file_path = './evaluationT/wer/evalute'
        if is_train:
            file_name = f"output-hypothesis-dev.ctm"
        else:
            file_name = f"output-hypothesis-test.ctm"
        file_path = os.path.join(source_file_path, file_name)
    # 预处理语言数据
    word2idx, word_number, idx2word = DataPreprocessing.word2id(train_label_path=train_label_path, valid_label_path=valid_label_path, test_label_path=test_label_path, data_set_name=data_set_name)
    # 训练时的数据预处理操作
    train_transform=VideoEnhancement.Compose([
        VideoEnhancement.RandomCrop(size=224),
        VideoEnhancement.RandomHorizontalFlip(prob=0.5),
        VideoEnhancement.ToTensor(),
        VideoEnhancement.TemporalRescale(temp_scaling=0.2)
    ])
    # 测试时的数据预处理操作
    test_transform=VideoEnhancement.Compose([
        VideoEnhancement.CenterCrop(size=224),
        VideoEnhancement.ToTensor(),
    ])
    # 导入数据
    train_data=DataPreprocessing.BaseSignLanguageDataset(image_dir_path=train_data_path,label_path=train_label_path,word2idx=word2idx,data_set_name=data_set_name,is_train=is_train,transform=train_transform)
    test_data=DataPreprocessing.BaseSignLanguageDataset(image_dir_path=test_data_path,label_path=test_label_path,word2idx=word2idx,data_set_name=data_set_name,is_train=is_train,transform=test_transform)
    valid_data=DataPreprocessing.BaseSignLanguageDataset(image_dir_path=valid_data_path,label_path=valid_label_path,word2idx=word2idx,data_set_name=data_set_name,is_train=is_train,transform=test_transform)
    # 构造DataLoader
    train_loader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=pin_memory,collate_fn=DataPreprocessing.collate_fn,drop_last=True)
    test_loader=DataLoader(dataset=test_data,batch_size=1,shuffle=False,num_workers=num_workers,pin_memory=pin_memory,collate_fn=DataPreprocessing.collate_fn,drop_last=True)
    valid_loader=DataLoader(dataset=valid_data,batch_size=1,shuffle=False,num_workers=num_workers,pin_memory=pin_memory,collate_fn=DataPreprocessing.collate_fn,drop_last=True)
    # 模型定义
    model=Net.ModuleNet(hidden_size=hidden_size,word_set_num=word_number*max_num_states+1,module_choice=module_choice,data_set_name=data_set_name,is_flag=True)
    model=model.to(device=device)
    # 定义损失函数
    ctc_loss,kld,mean_loss=loss_function(module_choice)
    loft_soft_max=nn.LogSoftmax(dim=-1)
    optimizer=optimizer_function(module=model,learning_rate=lr,weight_decay=0.0001)
    # 读取预训练模型参数
    best_loss=math.inf
    best_loss_epoch=0
    best_wer_score=math.inf
    best_wer_score_epoch=0
    epoch=0 # 当前完成的epoch
    last_epoch=-1 # 用于告诉学习率调度器目前已经完成了多少个epoch的训练
    if os.path.exists(current_module_path):
        checkpoint=torch.load(f=current_module_path,map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["module_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_loss=checkpoint["best_loss"]
        best_loss_epoch=checkpoint["best_loss_epoch"]
        best_wer_score=checkpoint["best_wer_score"]
        best_wer_score_epoch=checkpoint["best_wer_score_epoch"]
        epoch=checkpoint["epoch"]
        last_epoch=epoch
        # 打印已加载模型的状态
        logger.info(f"已加载预训练模型 epoch: {epoch}, best loss: {best_loss}, best loss epoch: {best_loss_epoch}, wer score: {best_wer_score}, best wer score epoch: {best_wer_score_epoch}")
    else:
        logger.info(f"已加载预训练模型 epoch: {epoch}, best loss: {best_loss}, best loss epoch: {best_loss_epoch}, wer score: {best_wer_score}, best wer score epoch: {best_wer_score_epoch}")
    # 设置学习率削减规则
    scheduler=optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=[35,45],
        gamma=0.2,
        last_epoch=last_epoch # 设置这个参数是为了契合模型续训练
    )
    # 解码参数
    decoder=decode.Decode(gloss_dict=word2idx,num_classes=word_number+1,search_mode="beam")
    # 训练
    pass
if __name__=="__main__":
    config_params=readconfig.read_config()
    train(config_params)