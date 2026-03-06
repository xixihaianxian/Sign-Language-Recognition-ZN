import torch
import readconfig
from loguru import logger
from typing import Tuple,Dict,Any,List
from torch import nn
import Net
import DataPreprocessing
import os
import VideoEnhancement
from tqdm import tqdm
from torch.utils.data import DataLoader
from train import loss_function
import numpy as np
import decode
from WER import wer_score
import random
from glob import glob
import imageio
import cv2
import pandas as pd
from torchvision import transforms

# 获取配置文件
config_params=readconfig.read_config()

# 模型结构参数
hidden_size=int(config_params["hiddenSize"])
max_num_states=1
num_workers=int(config_params["numWorkers"])
pin_memory=bool(config_params["pinMemory"])

# 默认设备
default_device=config_params["device"]

# 学习率
learning_rate=float(config_params["lr"])

# 获取train和test上的数据转化
def load_transform(is_train:bool=True):
    r"""
    :params:
        is_train: 是否是测试集
    :return:
        transform
    """
    if is_train==True:
        transform = VideoEnhancement.Compose([
            VideoEnhancement.RandomCrop(size=224),
            VideoEnhancement.RandomHorizontalFlip(prob=0.5),
            VideoEnhancement.ToTensor(),
            VideoEnhancement.TemporalRescale(temp_scaling=0.2)
        ])
    else:
        transform = VideoEnhancement.Compose([
            VideoEnhancement.CenterCrop(size=224),
            VideoEnhancement.ToTensor(),
        ])
    return transform

# 获取一些重要的参数
def fetch_data_params()->Tuple[str,str,str,str,str,str]:
    r"""
    :return: test_data_path,test_label_path,module_choice,data_set_name
    """
    # 数据路径
    test_data_path=config_params["testDataPath"]
    # 标签路径
    test_label_path=config_params["testLabelPath"]
    train_label_path=config_params["trainLabelPath"]
    valid_label_path=config_params["validLabelPath"]
    # 模型选择
    module_choice=config_params["moduleChoice"]
    # 数据集名称
    data_set_name=config_params["dataSetName"]
    return test_data_path,train_label_path,valid_label_path,test_label_path,module_choice,data_set_name

# 对数据进行一些预处理
def data_processing(train_label_path,valid_label_path,test_label_path,data_set_name):
    r"""
    :params:
        train_label_path: 训练集标签路径
        valid_label_path: 验证集标签路径
        test_label_path: 测试集标签路径
        data_set_name: 数据集名称
    :return:
        word2idx,word_number,idx2word
    """
    word2idx, word_number, idx2word = DataPreprocessing.word2id(train_label_path=train_label_path,
                                                                valid_label_path=valid_label_path,
                                                                test_label_path=test_label_path,
                                                                data_set_name=data_set_name)
    return word2idx,word_number,idx2word

# 登录最佳的模型
def load_module(module_choice:str,word_number:int,data_set_name:str,device=None,module_path:str="module/best.pth"):
    r"""
    :params:
        module_choice: 模型选择
        module_path: 存放模型参数的路径
    :return:
        module:best module
    """
    logger.info(f"Using the {module_choice} model!")
    module:nn.Module = Net.ModuleNet(hidden_size=hidden_size, word_set_num=word_number * max_num_states + 1,
                          module_choice=module_choice, data_set_name=data_set_name, is_flag=True,device=device if device is not None else default_device,module_dir="resnet")
    if not os.path.exists(module_path):
        logger.error(f"Not exists {module_path}, please check {module_path}!")
        exit(1)
    module_dict:Dict[str,Any]=torch.load(module_path,map_location=default_device if device is None else device,weights_only=False)
    module.load_state_dict(module_dict["module_state_dict"])
    return module

# 获取数据加载器
def get_data_loader(data_set_name:str,data_path:str,label_path,word2idx,is_train=False,batch_size=1,shuffle:bool=True):
    r"""
    :params:
        data_set_name: 数据集名称
        data_path: 数据路径
        label_path: 标签路径
        word2idx: word到id的映射
        is_train: 是否是训练
        batch_size: 批量大小
        shuffle: 是否打乱顺序
    :return:
        loader
    """
    transform=load_transform(is_train)
    class_name=data_set_name.replace("-","")
    method=getattr(DataPreprocessing,f"{class_name}Dataset")
    test_data = method(image_dir_path=data_path, label_path=label_path, word2idx=word2idx,
                       data_set_name=data_set_name, is_train=is_train, transform=transform)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                             pin_memory=pin_memory, collate_fn=DataPreprocessing.collate_fn, drop_last=True)
    return test_loader

# 测试
def test(idx2word:List[str],decoder,module:nn.Module,test_loader:DataLoader,device=None,module_choice:str="TFNet",data_set_name:str="CE-CSL"):
    r"""
    :params:
        idx2word:id到word的映射
        decoder:解码器
        module:模型
        test_loader:测试使用的数据加载器
        device:设置的设备
        module_choice:模型选择
        data_set_name:数据集选择
    """
    # 定义损失函数
    ctc_loss, kld, mse_loss = loss_function(module_choice)
    log_soft_max=nn.LogSoftmax(dim=-1)
    # 确定设备
    device=default_device if device is None else device
    # wer sum
    wer_score_sum = 0
    # 记录损失
    loss_values=list()
    module=module.to(device=device)
    # 将模型转化为测试模式
    module.eval()
    # 加载数据
    with torch.no_grad():
        for test_data in tqdm(test_loader):
            test_vido=test_data["video"].to(device=device)
            test_label:List[torch.Tensor]=test_data["label"]
            test_video_length=test_data["video_length"]
            info=test_data["info"]
            test_target_out_data=[label.to(device=device) for label in test_label]
            test_target_len=torch.tensor(list(map(len,test_target_out_data)))
            test_target_data=test_target_out_data
            test_target_out_data=torch.cat(test_target_out_data,dim=0).to(device=device)
            batch_size=len(test_target_len)
            log_probs_1, log_probs_2, log_probs_3, log_probs_4, log_probs_5, length, out_data_1, out_data_2, out_data_3 = module(test_vido, test_video_length, is_train=False)
            log_probs_1=log_soft_max(log_probs_1)
            if module_choice=="MSTNet":
                loss_1=ctc_loss(log_probs_1,test_target_out_data,length,test_target_len)
            else:
                loss_1=ctc_loss(log_probs_1,test_target_out_data,length,test_target_len).mean()
            loss=loss_1
            if np.isnan(loss.item()) or np.isinf(loss.item()):
                logger.warning("There is a problem with the loss value!")
                continue
            else:
                loss_values.append(loss.item())
            pred,test_target_data_ctc=decoder.decode(ctc_logits=log_probs_1,vid_lgt=length,batch_first=False,is_probability_distribution=False)
            # 对于RWTH和RWTH-T数据集，我们这里不深入分讨论
            if data_set_name=="CSL-Daily" or data_set_name=="CE-CSL":
                wer=wer_score(prediction_result=test_target_data_ctc,target_out_result=test_target_data,id2word=idx2word,batch_size=batch_size)
                wer_score_sum+=wer
        test_loss=np.mean(loss_values)
        test_wer=wer_score_sum/len(test_loader)
        logger.info(f"test loss is {test_loss:.2f}!")
        logger.info(f"test wer score is {test_wer:.2f}!")

def translation(idx2word:List[str],module:nn.Module,decoder,video_path:str=None,device:str=None,data_set_name:str="CE-CSL",transform=None):
    log_soft_max = nn.LogSoftmax(dim=-1)
    sentence=list()
    videos_path=os.path.join(data_set_name,"video","train")
    translator_dir=glob(f"{videos_path}/*")
    if device is None:
        device=default_device
    if video_path is None:
        translator=random.choice(translator_dir)
        video_path=random.choice(glob(f"{translator}/*"))
    # 创建存放图片的文件夹
    if not os.path.exists("example"):
        os.makedirs("example")
    video_file_name = os.path.splitext(os.path.basename(video_path))[0] # 抽取到的视频名称
    # 获取真实的标签
    video_dirname = os.path.dirname(video_path)
    if data_set_name in video_dirname:
        if video_path is None:
            data_type = os.path.split(videos_path)[-1]
        else:
            data_type=video_file_name.split("-")[0]
        label_path = os.path.join(data_set_name, 'label', f"{data_type}.csv")
        data=pd.read_csv(label_path)
        real_result=data[data["Number"]==video_file_name]
        real_label=real_result["Chinese Sentences"].item()
    else:
        real_label=None
    save_image_dir = os.path.join("example", video_file_name)
    os.makedirs(save_image_dir, exist_ok=True)
    video=imageio.get_reader(video_path) # 获取视频的信息
    nframe=video.count_frames()
    logger.info(f"The total number of frames in the test video is {nframe}.")
    try:
        for frame_number in range(nframe):
            frame_image=video.get_data(frame_number)
            frame_image=cv2.cvtColor(frame_image,code=cv2.COLOR_BGR2RGB)
            frame_image=cv2.resize(frame_image,dsize=(255,255))
            image_file_name=f"{frame_number:0>5}.jpg"
            image_path=os.path.join(save_image_dir,image_file_name)
            cv2.imencode(ext=".jpg", img=frame_image)[1].tofile(image_path)
        logger.info(f"{os.path.basename(video_path)} video processing completed")
    except Exception as error:
        logger.error(f"{os.path.basename(video_path)} video processing failure")
        raise Exception(f"{os.path.basename(video_path)} video processing failure")
    image_path_list=glob(os.path.join(f"{save_image_dir}/*"))
    image_number=len(image_path_list)
    indices=np.linspace(start=0,stop=image_number-1,num=image_number,dtype=np.int64)
    frames=[image_path_list[index] for index in indices]
    image_seq = [cv2.resize(cv2.cvtColor(cv2.imread(image_path), code=cv2.COLOR_BGR2RGB), dsize=(256, 256)) for image_path in frames]
    if transform is not None:
        image_seq=transform(image_seq)
    if isinstance(image_seq, list):
        image_seq=np.array(image_seq)
        image_seq=torch.tensor(image_seq,dtype=torch.float32)
    # 图片的标准化
    image_seq=image_seq.to(dtype=torch.float32)/255.0
    normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_seq=normalize(image_seq)
    image_seq_length=[torch.tensor([len(image_seq)])]
    image_seq=image_seq.unsqueeze(0)
    module=module.to(device=device)
    module.eval()
    with torch.no_grad():
        image_seq=image_seq.to(device=device)
        log_probs_1, log_probs_2, log_probs_3, log_probs_4, log_probs_5, length, out_data_1, out_data_2, out_data_3 = module(image_seq, image_seq_length, is_train=False)
        log_probs_1 = log_soft_max(log_probs_1)
        if decoder.search_mode=="beam":
            pred, target_data_ctc = decoder.decode(ctc_logits=log_probs_1, vid_lgt=length, batch_first=False, is_probability_distribution=False)
        else:
            pred = decoder.decode(ctc_logits=log_probs_1, vid_lgt=length, batch_first=False)
            sentences = [list(zip(*result))[0] for result in pred]
        if decoder.search_mode=="beam":
            for targets in target_data_ctc:
                for target in targets:
                    if isinstance(target,torch.Tensor):
                        sentence.append(idx2word[target.item()])
                    elif isinstance(target,int):
                        sentence.append(idx2word[target])
                    else:
                        logger.error(f"target type is {type(target)}")
                print(f"The translation result is :{''.join(sentence)}")
        else:
            for sentence in sentences:
                print(f"The translation result is :{''.join(sentence)}")
    if real_label is not None:
        print(f"The real translation result is :{real_label}")

if __name__=="__main__":
    test_data_path, train_label_path, valid_label_path, test_label_path, module_choice, data_set_name = fetch_data_params()
    word2idx, word_number, idx2word=data_processing(train_label_path, valid_label_path, test_label_path, data_set_name)
    # 构造解码器
    decoder= decode.Decode(gloss_dict=word2idx, num_classes=word_number + 1, search_mode="beam")
    # 模型
    module=load_module(module_choice=module_choice, word_number=word_number, data_set_name=data_set_name, module_path="module/MSTNet-Current.pth")
    # 数据生成器
    test_loader=get_data_loader(data_set_name=data_set_name,data_path=test_data_path,label_path=test_label_path,word2idx=word2idx,is_train=False)
    # 测试
    # test(idx2word=idx2word,decoder=decoder,module=module,test_loader=test_loader)
    # 翻译
    translation(idx2word, module, decoder,video_path=None, device=None, data_set_name=data_set_name, transform=load_transform(is_train=False))