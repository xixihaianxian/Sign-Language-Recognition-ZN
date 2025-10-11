import pickle
from typing import List,Dict,Tuple,Any
import config
import pandas as pd
import csv
from loguru import logger
from torch.utils import data
from glob import glob
import os
import numpy as np
import cv2
import torch
from collections import defaultdict

# 判断文件状态
def check_param_status(**kwargs):
    for key,value in kwargs.items():
        if value is None:
            logger.warning(f"{key}not set!")
            return True
    return False
# 删除非法字符
def remove_illegal_char(word:str)->str:
    # 定义栈，存放非法字符
    stack=list()
    # 存放结果
    result_chars=list()
    # 便利每一个字符
    for char in word:
        # 判断是不是开符号
        if char in config.OPEN_CHAR:
            stack.append(char) # 如果是开符号，加入栈
        elif char in config.CLOSE_CHAR_MAP.keys(): # 判断是不是闭符号
            # 判断栈最后一个符合是否和闭符号匹配
            if stack and stack[-1]==config.CLOSE_CHAR_MAP.get(char):
                stack.pop() # 匹配，删除stack里面最后一个符号
            else:
                pass
        # 将合法的符号加入result_chars
        else:
            if not stack:
                result_chars.append(char)
    # 返回结果
    return "".join(result_chars)
# 预处理words
def handle_words(words:List[str])->List[str]:
    r"""
    处理words里面的word
    """
    # 遍历每一个word
    for i in range(len(words)):
        word=words[i]
        # 删除word不必要的部分
        word=remove_illegal_char(word)
        # 删除不必要的数字
        if word[-1].isdigit():
            if not word[0].isdigit():
                word=word[:-1]
        # 统一符号
        if word[0]=="," or word[0]=="，": #TODO 如何使用分词的话这里需要进行相应的修改
            word="，"+word[1:]
        if word[0]=="?" or word[0]=="？":
            word="？"+word[1:]
        # if word=="," or word=="，":
        #     word="，"
        # if word=="?" or word=="？":
        #     word="？"
        # 去掉数字前导零
        if word.isdigit():
            word=str(int(word))
        words[i]=word
    return words
# 建word到id的映射
def word2id(train_label_path:str=None,valid_label_path:str=None,test_label_path:str=None,data_set_name:str="CSL-Daily"):
    r"""
    train_label_path: 训练标签文件地址
    valid_label_path: 验证标签文件地址
    test_label_path: 测试标签文件地址
    data_set_name: 数据集名称（默认为CSL-Daily）
    """
    PAD="<pad>"
    # 定义word list
    word_list = list()
    # label路径列表
    label_paths=[train_label_path, test_label_path, valid_label_path]
    # RWTH数据处理方式，可以用来微调中文模型
    if data_set_name=="RWTH":
        # 处理train label
        if not check_param_status(train_label_path=train_label_path):
            train_df=pd.read_csv(train_label_path,sep="|")
            train_annotations=train_df.loc[:,"annotation"]
            # 遍历每一句
            for annotation in train_annotations:
                words=annotation.split()
                word_list.extend(words)
        # 处理test label
        if not check_param_status(test_label_path=test_label_path):
            test_df=pd.read_csv(test_label_path,sep="|")
            test_annotations=test_df.loc[:,"annotation"]
            for annotation in test_annotations:
                words=annotation.split()
                word_list.extend(words)
        # 处理valid label
        if not check_param_status(valid_label_path=valid_label_path):
            valid_df=pd.read_csv(valid_label_path,sep="|")
            valid_annotations=valid_df.loc[:,"annotation"]
            for annotation in valid_annotations:
                words=annotation.split()
                word_list.extend(words)
    # 处理CE-CSL
    elif data_set_name=="CE-CSL":
        # 处理train data
        if not check_param_status(train_label_path=train_label_path):
            train_df=pd.read_csv(train_label_path,sep=",")
            train_gloss=train_df.loc[:,"Gloss"]
            for gloss in train_gloss:
                words=gloss.split("/")
                words=handle_words(words)
                word_list.extend(words)
        # 处理valid data
        if not check_param_status(valid_label_path=valid_label_path):
            valid_df=pd.read_csv(valid_label_path,sep=",")
            valid_gloss=valid_df[:,"Gloss"]
            for gloss in valid_gloss:
                words=gloss.split("/")
                words=handle_words(words)
                word_list.append(words)
        # 处理test data
        if not check_param_status(test_label_path=test_label_path):
            test_df=pd.read_csv(test_label_path,sep=",")
            test_gloss=test_df[:,"Gloss"]
            for gloss in test_gloss:
                words=gloss.split("/")
                words=handle_words(words)
                word_list.append(words)
    # 处理RWTH-T
    elif data_set_name == "RWTH-T":
        label_paths_name = ["train_label_path", "test_label_path", "valid_label_path"]
        # 遍历所有label path
        for label_name,label_path in zip(label_paths_name,label_paths):
            # 判断参数状态
            if label_path is None:
                logger.warning(f"{label_name} not set!")
            else:
                # 建立上下文管理器，打开label path
                with open(label_path, 'r', encoding="utf-8") as file:
                    reader = csv.reader(file)
                    for index, row in enumerate(reader):
                        # 排除标签行
                        if index != 0:
                            row_str_list = row[0].split("|")
                            words =row_str_list[5].split()
                            word_list.extend(words)
    # 处理CSL-Daily
    elif data_set_name=="CSL-Daily":
        words_pkl="/mnt/e/Sign-Language-Recognition-ZN/labelData/CSL-Daily/csl2020ct_v2.pkl"
        with open(words_pkl,mode="rb") as pkl:
            information=pickle.load(pkl)
            word_list=handle_words(information["gloss_map"])
    # 构建word2idx和idx2word
    idx2word=[PAD] # 将pad加入到idx2word
    word_set=sorted(list(set(word_list)))
    idx2word.extend(word_set)
    word2idx=dict(set((word,index) for index,word in enumerate(idx2word)))
    # word_number=len(idx2word) # TODO len(word2id)比较合理
    word_number=len(idx2word)-1
    # 返回word2idx，词的数量，idx2word
    return word2idx,word_number,idx2word # 此时的id2word本质是word_list
# 构建基础Dataset
class BaseSignLanguageDataset(data.Dataset):
    def __init__(self,image_dir_path:str,label_path:str,word2idx:Dict[str,int],data_set_name:str,is_train:bool=False,transform=None):
        super().__init__()
        # 构建属性
        self.image_dir_path=image_dir_path # 存放图片的文件夹
        self.label_path=label_path # label文件路径
        self.word2dix=word2idx # word->idx
        self.data_set_name=data_set_name # 数据集名称
        self.is_train=is_train # 是否是训练
        self.transform=transform # 是否对图像进行增强
        self.samples=list() # 定义sample列表
    # 登录数据
    def load_data(self,image_dir_path,label_path):
        logger.error(f"The load data method is not defined.") # 日志：load data方法没有定义
        raise NotImplementedError(f"The load data method is not defined.")
    # 处理sentence数据
    def process_label(self,text:Any):
        if isinstance(text,list):
            words=text
        elif isinstance(text,str):
            words=text.split()
        else:
            logger.error(f"Data of type {type(text)} is not supported")
            raise ValueError(f"Data of type {type(text)} is not supported")
        text_to_id=list()
        for word in words:
            try:
                text_to_id.append(self.word2dix[word]) # TODO 一般来说是一定会存在的，因为我构建词集的时候是将所有的词都加入到word_list里面的
            except KeyError:
                raise KeyError(f"The {word} key does not exist")
        return text_to_id
    # 收集帧索引
    def sample_indices(self,n:int):
        indices=np.linspace(start=0,stop=n-1,num=int(n),dtype=np.int64)
        return indices
    # 获取图片序列
    def load_frame(self,image_seq_path):
        image_path_list=glob(os.path.join(image_seq_path,"*")) # 获取目标目录下的所有图片文件
        image_number=len(image_path_list)
        indices=self.sample_indices(image_number) # 获取关键帧索引
        frames=[image_path_list[index] for index in indices] # 获取关键帧，但是这里是获取所有帧，如果去需要获取关键帧，可以在image_number上做更改
        # 此时image的大小还是(256,256,3)
        image_seq=[cv2.resize(cv2.cvtColor(cv2.imread(image_path),code=cv2.COLOR_BGR2RGB),dsize=(256,256)) for image_path in image_path_list]
        # 对图片进行图像增强
        if self.transform is not None:
            image_seq=self.transform()
        image_seq=image_seq.to(dtype=torch.float32)/127.5 - 1
        return image_seq
    def __getitem__(self,item):
        image_seq_path,label=self.samples[item]
        info=os.path.basename(image_seq_path) # 视频名称
        image_seq=self.load_frame(image_seq_path) # 视频总帧
        sample={"video":image_seq,"label":label,"info":info}
        return sample
    def __len__(self):
        return len(self.samples)
# 构建RWTHDataset
class RWTHDataset(BaseSignLanguageDataset):
    def __init__(self,image_dir_path:str,label_path:str,word2idx:Dict[str,int],data_set_name:str,is_train:bool=False,transform=None):
        super().__init__(image_dir_path,label_path, word2idx, data_set_name, is_train, transform)
    def load_data(self,image_dir_path,label_path):
        # 定义label_dict
        label_dict=defaultdict(str)
        df=pd.read_csv(label_path,sep="|")
        label_dict=dict(zip(df.loc[:,"id"].values,df.loc[:,"annotation"].values))
        # 定义label,存放{id:label},label是经过处理之后的
        labels=defaultdict(list)
        for key in label_dict.keys():
            labels[key]=self.process_label(label_dict.get(key))
        # 存放视频帧的所有目录
        files_name=os.listdir(image_dir_path)
        files_name=sorted(files_name)
        for name in files_name:
            try:
                image_seq_path=os.path.join(image_dir_path,name)
                self.samples.append((image_seq_path,labels[name]))
            except Exception  as error:
                logger.error(f"{name} does not exist")
                raise KeyError(f"{name} does not exist!")

if __name__=="__main__":
    word2idx,word_number,idx2word=word2id()
    print(word2idx)