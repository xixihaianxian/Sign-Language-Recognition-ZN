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
import math

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
        words_pkl=config.CSL_Daily_Data_PATH
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
    def process_label(self,text:Any,sep:str=None)->List[int]:
        if isinstance(text,list):
            words=text
        elif isinstance(text,str):
            if sep is None:
                words=text.split()
            else:
                words=text.split(sep)
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
    def __getitem__(self, item):
        image_seq_path,label=self.samples[item]
        info=os.path.basename(image_seq_path) # 获取视频的名称
        image_seq_path=os.path.join(image_seq_path,"1") # 获取主摄像机位置的video
        image_seq=self.load_frame(image_seq_path)
        sample={"video":image_seq,"label":label,"info":info}
        return sample
# 构造RWTHT的Dataset
class RWTHTDataset(RWTHDataset):
    def __init__(self,image_dir_path:str,label_path:str,word2idx:Dict[str,int],data_set_name:str,is_train:bool=False,transform=None):
        super().__init__(image_dir_path, label_path, word2idx, data_set_name, is_train, transform)
    def load_data(self,image_dir_path,label_path):
        label_dict=defaultdict(str)
        df=pd.read_csv(label_path,sep="|")
        for name,orth in zip(df.loc[:,"name"],df.loc[:,"orth"]):
            label_dict[name]=orth
        label=defaultdict(list)
        for name in label_dict:
            label[name]=self.process_label(label_dict[name])
        files_name=os.listdir(image_dir_path)
        for file_name in files_name:
            try:
                image_seq_path=os.path.join(image_dir_path,file_name)
                self.samples.append((image_seq_path,label[file_name]))
            except Exception as error:
                logger.error(f"{file_name} does not exist")
                raise KeyError(f"{file_name} does not exist")
class CECSLDataset(BaseSignLanguageDataset):
    def __init__(self,image_dir_path:str,label_path:str,word2idx:Dict[str,int],data_set_name:str,is_train:bool=False,transform=None):
        super().__init__(image_dir_path, label_path, word2idx, data_set_name, is_train, transform)
    def load_data(self,image_dir_path,label_path):
        label_dict=defaultdict(str)
        df=pd.read_csv(label_path,sep=",")
        for number,gloss in zip(df.loc[:,"Number"],df.loc[:,"Gloss"]):
            label_dict[number]=gloss
        label=defaultdict(list)
        for number in label_dict:
            label[number]=self.process_label(label_dict[number],sep="/")
        files_name=os.listdir(image_dir_path)
        for file_name in files_name:
            translator_dir=os.path.join(image_dir_path,file_name) # 这里不太一样，先获取翻译员的目录
            video_dir=sorted(os.listdir(translator_dir))
            for video in video_dir: # 视频目录，但如果是视频处理之后的dir的话就是frame文件目录
                try:
                    image_seq_path=os.path.join(translator_dir,video)
                    self.samples.append((image_seq_path,label[video]))
                except Exception as error:
                    logger.error(f"{video} does not exist")
                    raise KeyError(f"{video} does not exist")
class CSLDailyDataset(BaseSignLanguageDataset):
    def __init__(self,image_dir_path:str,label_path:str,word2idx:Dict[str,int],data_set_name:str,is_train:bool=False,transform=None):
        super().__init__(image_dir_path, label_path, word2idx, data_set_name, is_train, transform)
    def load_data(self,image_dir_path,label_path):
        label=defaultdict(list)
        with open(config.CSL_Daily_Data_PATH,"rb") as file:
            data=pickle.load(file)
        # 获取info字段
        info=data["info"]
        for item in info:
            gloss=item.get("label_gloss") # 使用get来获取gloss
            if gloss is not None: # 如果gloss不存在返回None
                raise KeyError(f"No gloss found about {item.get("name")}")
            else:
                label[item.get("name")]=self.process_label(gloss)
        df=pd.read_csv(label_path,sep="|",header=None,names=["dir_name","kind"]) # 可以使用read_csv来读取txt文件
        for image_seq_name in df.loc[:,"dir_name"]:
            image_seq_path=os.path.join(image_dir_path,image_seq_name)
            try:
                self.samples.append((image_seq_path,label[image_seq_name]))
            except KeyError as error:
                logger.error(f"{image_seq_name} is not exist")
                raise KeyError(f"{image_seq_name} is not exist")
# 定义默认字典
class custom_defaultdict(defaultdict):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs) # 继承父类
        self.warned=set()
        self.warning_enabled=False # 是否启用这个warning,默认是没有启用的
    def __getitem__(self,key):
        if key=="text" and key not in self.warned and self.warning_enabled:
            logger.warning(f"The batch['text'] is no longer in use, please replace it with a batch['label'].")
            # 将key更新到warned里面
            self.warned.add(key)
        # 使用父类的__getitem__来获取value
        return super().__getitem__(key)
# 设置collate_fn函数，可用于之后的DataLoader,同时处理样本
# 论文参考1. https://arxiv.org/pdf/2402.19118 2. https://arxiv.org/pdf/1910.06709 3. https://arxiv.org/pdf/2311.07623
def collate_fn(batch):
    collated=custom_defaultdict()
    # 对batch进行排序
    batch=list(sorted(batch,key=lambda x:len(x["video"]),reverse=True))
    # 获取视频时间最长的视频的总帧
    max_len=len(batch[0]["video"])
    # 左边补帧，为了保持时序上下文，让卷积或Transformer能在开头就有足够上下文窗口
    left_pad=6
    # 总步长，模型中多层Conv、Pooling、或Temporal Transformer导致的总下采样率（比如 4 表示长度被缩小 4 倍）
    total_stride=4
    # 右边补帧，使补完后的长度能被 total_stride 整除，避免特征图维度不匹配错误
    right_pad=math.ceil(max_len/total_stride)*total_stride-max_len+left_pad
    # 计算出新的总长
    max_len=max_len+left_pad+right_pad
    # 填充总帧
    modify_videos=list()
    for sample in batch:
        video=sample.get("video")
        # collated["video_length"].append(torch.tensor([math.ceil(len(video)/total_stride)*total_stride+left_pad+right_pad]))
        collated["video_length"].append(torch.tensor([math.ceil(len(video)/total_stride)*total_stride+2*left_pad]))
        modify_videos.append(
            torch.cat(
                (
                    video[0][None].expand(left_pad,-1,-1,-1), # 左边补6帧
                    video, # 中间不发生改变
                    video[-1][None].expand(max_len-len(video)-left_pad,-1,-1,-1) # 右边补帧
                    
                ),dim=0
            )
        )
        collated["label"].append(torch.tensor(sample.get("label"),dtype=torch.int64))
        collated["info"].append(sample.get("info"))
        collated["expand"].append([left_pad,max_len-left_pad-len(video)])
    modify_videos=torch.stack(modify_videos,dim=0)
    collated["video"]=modify_videos
    collated.warning_enabled=True # 启动warning
    return collated
# 这个好像用不到，主要功能是计算数据的长度，有多少条数据，以及将数据转移到指定的设备上
def data_reshape(seq_data,device):
    seq_data_len=list(map(len,seq_data))
    batchsize=len(seq_data_len)
    seq_data=torch.cat(seq_data,dim=0).to(device=torch.device(device))
    return seq_data,batchsize,seq_data_len
# 标签处理，CTC是针对模型输出标签的处理，并不是输入标签的处理。经过神经网络和CTC层后，包含重复和空白，需要remove_blank这样的后处理函数来清理
# https://www.cs.toronto.edu/~graves/icml_2006.pdf
def remove_blank(label,max_sentence_len,blank=0,pad=0):
    modify_label=list()
    previous=None # 上一次的标签
    # 去除连续相同的标签
    for item in label:
        if item!=previous:
            modify_label.append(item)
            previous=item
        else:
            pass
    # 删除blank
    modify_label=list(item for item in modify_label if item!=blank)
    # 使用pad来填充modify_label
    if len(modify_label)<max_sentence_len:
        len_lack=max_sentence_len-len(modify_label)
        modify_label.extend([pad]*len_lack)
    else:
        modify_label=modify_label[:max_sentence_len]
    # 转化为张量
    modify_label=torch.tensor(modify_label,dtype=torch.int64)
    return modify_label
def ctc_greedy_decode(result:torch.Tensor,max_sentence_len,pad,blank=0):
    # 去每列的最大值索引
    index=result.argmax(dim=-1)
    # 索引为0的位置，值为blank
    result=remove_blank(label=index,max_sentence_len=max_sentence_len,blank=blank,pad=pad)
    return result
# 将输出的结果写入文件
def write_to_file(path:str,info,output):
    with open(path,mode="w",encoding="utf-8") as file:
        for sample_index,sample in enumerate(output):
            for word_index,word in enumerate(sample):
                # info 里面包含了视频的名称
                # word_index*1.0/100获取的是单个词开始的时间戳
                # (word_index+1)*1.0/100获取的是单个词结束的时间戳
                # word 获取的单个单词
                file.writelines(f"{info[sample_index]} 1 {word_index*1.0/100:.2f} {(word_index+1)*1.0/100:.2f} {word[0]}")

if __name__=="__main__":
    word2idx,word_number,idx2word=word2id()
    print(word2idx)