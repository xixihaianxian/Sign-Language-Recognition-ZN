import numpy as np
import torch
import readconfig
from loguru import logger
from typing import Tuple,Dict,Any
from torch import nn
import Net
import DataPreprocessing

# 获取配置文件
config_params=readconfig.read_config()

# 模型结构参数
hidden_size=config_params["hiddenSize"]
max_num_states=1


def fetch_test_params()->Tuple[str,str,str,str,str,str]:
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

def data_processing(train_label_path,valid_label_path,test_label_path,data_set_name):
    word2idx, word_number, idx2word = DataPreprocessing.word2id(train_label_path=train_label_path,
                                                                valid_label_path=valid_label_path,
                                                                test_label_path=test_label_path,
                                                                data_set_name=data_set_name)
    return word2idx,word_number,idx2word

def load_module(module_choice:str,word_number:int,data_set_name:str,module_path:Dict[str,Any]="module/best.pth"):
    r"""
    :params:
        module_choice: 模型选择
        module_path: 存放模型参数的路径
    :return:
        best module
    """
    module = Net.ModuleNet(hidden_size=hidden_size, word_set_num=word_number * max_num_states + 1,
                          module_choice=module_choice, data_set_name=data_set_name, is_flag=True)
    module_dict=torch.load(module_path,map_location="cpu")

def test():
    pass