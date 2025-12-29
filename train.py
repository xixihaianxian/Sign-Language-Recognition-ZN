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
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from typing import List
from WER import wer_score

# 创建模型存放目录
if not os.path.exists("module"):
    os.makedirs("module")
epoch_path=os.path.join("module","epoch")
if not os.path.exists(epoch_path):
    os.makedirs(epoch_path)

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
    mse_loss=None
    if module_choice=="MSTNet":
        ctc_loss=nn.CTCLoss(blank=pad_idx,reduction="mean",zero_infinity=True)
    elif module_choice=="VAC" or module_choice=="CorrNet" or module_choice=="MAM-FSD" or module_choice=="SEN" or module_choice=="TFNet":
        ctc_loss=nn.CTCLoss(blank=pad_idx,reduction="none",zero_infinity=True)
        kld=DataPreprocessing.SeqKD(T=8)
        if module_choice=="MAM-FSD":
            mse_loss=nn.MSELoss(reduction="mean")
    return ctc_loss,kld,mse_loss

# 构造优化器
def optimizer_function(module:torch.nn.Module,learning_rate:float,weight_decay:float)->optim.Optimizer:
    return optim.Adam(module.parameters(),lr=learning_rate,weight_decay=weight_decay)

# 训练
def train(config_params:Dict[str,Any],is_train=True):
    # 初始化数据路径参数
    train_data_path=config_params["trainDataPath"]
    valid_data_path=config_params["validDataPath"]
    test_data_path=config_params["testDataPath"]
    # 初始化标签路径参数
    train_label_path=config_params["trainLabelPath"]
    valid_label_path=config_params["validLabelPath"]
    test_label_path=config_params["testLabelPath"]
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
    method_name=f"{data_set_name.replace('-','')}Dataset" # 方法的名称
    method=getattr(DataPreprocessing,method_name)
    train_data=method(image_dir_path=train_data_path,label_path=train_label_path,word2idx=word2idx,data_set_name=data_set_name,is_train=is_train,transform=train_transform)
    test_data=method(image_dir_path=test_data_path,label_path=test_label_path,word2idx=word2idx,data_set_name=data_set_name,is_train=is_train,transform=test_transform)
    valid_data=method(image_dir_path=valid_data_path,label_path=valid_label_path,word2idx=word2idx,data_set_name=data_set_name,is_train=is_train,transform=test_transform)
    # 构造DataLoader
    train_loader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=pin_memory,collate_fn=DataPreprocessing.collate_fn,drop_last=True)
    test_loader=DataLoader(dataset=test_data,batch_size=1,shuffle=False,num_workers=num_workers,pin_memory=pin_memory,collate_fn=DataPreprocessing.collate_fn,drop_last=True)
    valid_loader=DataLoader(dataset=valid_data,batch_size=1,shuffle=False,num_workers=num_workers,pin_memory=pin_memory,collate_fn=DataPreprocessing.collate_fn,drop_last=True)
    # 模型定义
    model=Net.ModuleNet(hidden_size=hidden_size,word_set_num=word_number*max_num_states+1,module_choice=module_choice,data_set_name=data_set_name,is_flag=True)
    model=model.to(device=device)
    # 定义损失函数
    ctc_loss,kld,mse_loss=loss_function(module_choice)
    log_soft_max=nn.LogSoftmax(dim=-1)
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
    if is_train:
        epoch_number=55
        if last_epoch!=-1:
            epoch_number=epoch_number-last_epoch
        else:
            epoch_number=epoch_number
        seed=1
        for _ in range(epoch_number):
            model.train() # 将模型转化为训练模式
            scaler=GradScaler() # 梯度缩放，防止梯度因为过小引起的报错
            loss_value=list() # 存放损失值
            for data in tqdm(stable(dataloader=train_loader,seed=seed+epoch)):
                video=data.get("video").to(device=device) # data
                label:List[torch.Tensor]=data.get("label")
                video_len=data.get("video_length") # data length
                target_data=[target.to(device=device) for target in label]
                target_len=torch.tensor(list(map(len,target_data)))
                target_data=torch.cat(target_data,dim=0).to(device=device)
                with autocast():
                    log_probs_1, log_probs_2, log_probs_3, log_probs_4, log_probs_5, length, out_data_1, out_data_2, out_data_3=model(video,video_len,True)
                    # log_probs_1：Transformer编码后，最低T/4时，语义级对齐
                    # log_probs_2：第二卷积之后，T/4，结构时序建模
                    # log_probs_3：第一组卷积之后，T/2，中间监督防止，梯度消失
                    # log_probs_4：ResNet后线性映射，T，保留细节，辅助定位
                    if module_choice=="MSTNet":
                        log_probs_1=log_soft_max(log_probs_1)
                        log_probs_2=log_soft_max(log_probs_2)
                        log_probs_3=log_soft_max(log_probs_3)
                        log_probs_4=log_soft_max(log_probs_4)
                        loss_1=ctc_loss(log_probs_1,target_data,length,target_len)
                        loss_2=ctc_loss(log_probs_2,target_data,length,target_len)
                        loss_3=ctc_loss(log_probs_3,target_data,length*2,target_len)
                        loss_4=ctc_loss(log_probs_4,target_data,length*4,target_len)
                        loss:torch.Tensor=loss_1+loss_2+loss_3+loss_4
                    elif module_choice=="VAC" or module_choice=="CorrNet" or module_choice=="MAM-FSD" or module_choice=="SEN" or module_choice=="TFNet":
                        loss_3=25*kld(log_probs_2,log_probs_1,use_blank=False)
                        log_probs_1=log_soft_max(log_probs_1)
                        log_probs_2=log_soft_max(log_probs_2)
                        loss_1=ctc_loss(log_probs_1,target_data,length,target_len).mean()
                        loss_2 = ctc_loss(log_probs_2, target_data, length, target_len).mean()
                        if module_choice=="MAM-FSD":
                            loss_4 = mse_loss(out_data_1[0], out_data_1[1])
                            loss_5 = mse_loss(out_data_2[0], out_data_2[1])
                            loss_6 = mse_loss(out_data_3[0], out_data_3[1])
                            loss:torch.Tensor = loss_1 + loss_2 + loss_3 + 5 * loss_4 + 1 * loss_5 + 70 * loss_6
                        elif module_choice=="TFNet":
                            loss_6 = 25 * kld(log_probs_4, log_probs_3, use_blank=False)
                            log_probs_3 = log_soft_max(log_probs_3)
                            log_probs_4 = log_soft_max(log_probs_4)
                            loss_4 = ctc_loss(log_probs_3, target_data, length, target_len).mean()
                            loss_5 = ctc_loss(log_probs_4, target_data, length, target_len).mean()
                            log_probs_5 = log_soft_max(log_probs_5)
                            loss_7 = ctc_loss(log_probs_5, target_data, length, target_len).mean()
                            loss:torch.Tensor = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7
                        else:
                            loss:torch.Tensor = loss_1 + loss_2 + loss_3
                    if np.isinf(loss.item()) or np.isnan(loss.item()):
                        logger.error(f"loss is nan or inf!")
                        # raise ValueError(f"loss is nan or inf!")
                        continue
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                loss_value.append(loss.item())
                torch.cuda.empty_cache() # 释放 PyTorch 内部缓存的空闲显存(可能会降低性能)
            logger.info(f"epoch: {epoch} train loss: {np.mean(loss_value)} learning rate: {optimizer.param_groups[0]['lr']}")
            epoch+=1
            scheduler.step()
            # 模型验证
            with torch.no_grad():
                model.eval() # 模型转化为测试模式
                logger.info(f"model valid!")
                wer_score_sum = 0 # 记录此错误率
                total_info = [] # 存储每条样本的元信息
                total_sent = [] # 存储每个样本的模型预测结果
                valid_loss_value = [] # 收集每个batch的损失值
                for valid_data in tqdm(valid_loader):
                    valid_video=valid_data["video"].to(device=device)
                    valid_label:List[torch.Tensor]=valid_data["label"]
                    valid_video_length=valid_data["video_length"]
                    info=valid_data["info"]
                    valid_target_out_data=[target.to(device=device) for target in valid_label]
                    valid_target_len=torch.tensor(list(map(len,valid_target_out_data)))
                    valid_target_data=valid_target_out_data
                    valid_target_out_data=torch.cat(valid_target_out_data,dim=0).to(device=device)
                    batch_size=len(valid_target_len)
                    log_probs_1, log_probs_2, log_probs_3, log_probs_4, log_probs_5, length, out_data_1, out_data_2, out_data_3 = model(valid_video, valid_video_length, False)
                    log_probs_1=log_soft_max(log_probs_1)
                    if module_choice=="MSTNet":
                        loss_1=ctc_loss(log_probs_1,valid_target_out_data,length,valid_target_len)
                    else:
                        loss_1=ctc_loss(log_probs_1,valid_target_out_data,length,valid_target_len).mean()
                    loss=loss_1
                    if np.isnan(loss.item()) or np.isinf(loss.item()):
                        logger.error(f"loss is nan!")
                        continue
                    else:
                        valid_loss_value.append(loss.item())
                    pred,valid_target_data_ctc=decoder.decode(ctc_logits=log_probs_1,vid_lgt=length,batch_first=False,is_probability_distribution=False)
                    if data_set_name=="RWTH" or data_set_name=="RWTH-T":
                        total_info.extend(info)
                        total_sent+=pred
                    elif data_set_name=="CSL-Daily" or data_set_name=="CE-CSL":
                        wer=wer_score(prediction_result=[valid_target_data_ctc],target_out_result=valid_target_data,id2word=idx2word,batch_size=batch_size)
                        wer_score_sum+=wer
                torch.cuda.empty_cache()
                current_loss=np.mean(valid_loss_value)
                wer=wer_score_sum/len(valid_loader)
                if wer<best_wer_score:
                    best_wer_score=wer
                    best_wer_score_epoch=epoch-1
                    module_dict=dict()
                    module_dict["module_state_dict"]=model.state_dict()
                    module_dict["optimizer_state_dict"]=optimizer.state_dict()
                    module_dict["best_loss"]=best_loss # 这里可能有一些歧义其实也可以直接设置为上一次的current_loss，因为确定wer_score最佳的时候无法确定best_loss是最佳的
                    module_dict["best_loss_epoch"]=best_loss_epoch # 这里和上面的情况同理，可以直接设置为上一次的best_loss_epoch
                    module_dict["best_wer_score"]=best_wer_score
                    module_dict["best_wer_score_epoch"]=best_wer_score_epoch
                    module_dict["epoch"]=epoch
                    torch.save(module_dict,best_module_path)
                    logger.info(f"Save best module!")
                if best_loss>current_loss:
                    best_loss = current_loss
                    best_loss_epoch = epoch - 1
                    module_dict["module_state_dict"]=model.state_dict()
                    module_dict["optimizer_state_dict"]=optimizer.state_dict()
                    module_dict["best_loss"]=best_loss
                    module_dict["best_loss_epoch"]=best_loss_epoch
                    module_dict["best_wer_score"]=best_wer_score
                    module_dict["best_wer_score_epoch"]=best_wer_score_epoch
                    module_dict["epoch"]=epoch
                    torch.save(module_dict,current_module_path)
                    logger.info(f"Save current module!")
                # 保存每次epoch的模型
                epoch_module_save_path=os.path.join(epoch_path,f"Epoch_{epoch}_Module.pth")
                torch.save(module_choice,epoch_module_save_path)
                logger.info(f"valid loss: {current_loss} wer score: {wer}.")
                logger.info(f"best loss: {best_loss} best loss epoch: {best_loss_epoch} best wer score: {best_wer_score} best wer score epoch: {best_wer_score_epoch}.")

if __name__=="__main__":
    config_params=readconfig.read_config()
    train(config_params)