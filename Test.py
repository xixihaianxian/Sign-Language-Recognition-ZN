import math
import torch
import readconfig
from loguru import logger
from typing import Tuple,Dict,Any
from torch import nn
import Net
import DataPreprocessing
import os
import VideoEnhancement
from torch.utils.data import DataLoader

# 获取配置文件
config_params=readconfig.read_config()

# 模型结构参数
hidden_size=config_params["hiddenSize"]
max_num_states=1
num_workers=config_params["numWorkers"]
pin_memory=config_params["pinMemory"]

# 默认设备
default_device=config_params["device"]

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
def load_module(module_choice:str,word_number:int,data_set_name:str,device:None,module_path:str="module/best.pth"):
    r"""
    :params:
        module_choice: 模型选择
        module_path: 存放模型参数的路径
    :return:
        module:best module
    """
    module:nn.Module = Net.ModuleNet(hidden_size=hidden_size, word_set_num=word_number * max_num_states + 1,
                          module_choice=module_choice, data_set_name=data_set_name, is_flag=True,device=device if device is not None else default_device)
    if not os.path.exists(module_path):
        logger.error(f"Not exists {module_path}, please check {module_path}!")
        exit(1)
    module_dict:Dict[str,Any]=torch.load(module_path,map_location=default_device if device is None else device)
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
    method=getattr(DataPreprocessing,class_name)
    test_data = method(image_dir_path=data_path, label_path=label_path, word2idx=word2idx,
                       data_set_name=data_set_name, is_train=is_train, transform=transform)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                             pin_memory=pin_memory, collate_fn=DataPreprocessing.collate_fn, drop_last=True)
    return test_loader

# 测试
def test(module:nn.Module,test_loader:DataLoader):
    # 最低的wer
    best_wer_score=math.inf
    # 最佳的损失
    best_loss=math.inf

if __name__=="__main__":
    test_data_path, train_label_path, valid_label_path, test_label_path, module_choice, data_set_name = fetch_data_params()
    word2idx, word_number, idx2word=data_processing(train_label_path, valid_label_path, test_label_path, data_set_name)
    print(word_number)