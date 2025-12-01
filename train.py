import torch
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from typing import Dict
import readconfig
import DataPreprocessing

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

# 训练
def train(config_params:Dict[str,str],is_train=True):
    # 初始化数据路径参数
    trainDatPath=config_params["trainDataPath"]
    validDataPath=config_params["validDataPath"]
    testDataPath=config_params["testDataPath"]
    # 初始化标签路径参数
    trainLabelPath=config_params["trainLabelPath"]
    validLabelPath=config_params["validLabelPath"]
    testLabelPath=config_params["testLabelPath"]
    # 初始化模型位置参数
    best_module_path = config_params["bestModuleSavePath"]
    current_module_path = config_params["currentModuleSavePath"]
    # 初始化参数
    device = config_params["device"]
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
    word2id, word_number, id2word = DataPreprocessing.word2id(train_label_path=trainLabelPath, valid_label_path=validLabelPath, test_label_path=testLabelPath, data_set_name=data_set_name)
if __name__=="__main__":
    config_params=readconfig.read_config()
    train(config_params)