import torch
from itertools import groupby
import ctcdecode
from torch.nn import functional as F
from six.moves import xrange
from torch import nn
from collections import defaultdict
from typing import Dict
from loguru import logger

# 负责把模型输出(CTC logits)转化为可读的序列
class Decode:
    def __init__(self,gloss_dict:Dict[str,int],num_classes:int,search_mode:str,blank_id=0):
        r"""
        gloss_dict: gloss到标签的字典
        num_classes：类别总数
        search_model：解码方式
        blank_id：空白符的id
        """
        self.gloss_dict=gloss_dict # gloss到标签的字典
        self.num_classes=num_classes # 总类别数
        self.search_mode=search_mode # 控制解码方式（beam,max）
        self.blank_id=blank_id # CTC空白符号的id
        self.gloss2id=defaultdict(int)
        for gloss,idx in gloss_dict.items():
            if idx == 0:
                continue
            self.gloss2id[gloss]=idx
        self.id2gloss=dict([(idx,gloss) for gloss,idx in zip(self.gloss2id.keys(),self.gloss2id.values())])
        # 构造虚拟词表
        vocab=[chr(number) for number in range(2000,2000+self.num_classes)]
        # 实现Beam search 解码
        self.decoder=ctcdecode.CTCBeamDecoder(
            labels=vocab, # 所有类别的标签列表，用于将解码出的index转化为字符
            beam_width=10, # 控制候选路径数，越大越准越慢
            blank_id=blank_id, # blank索引
            num_processes=10,# 并行处理进程数
        )
    # 解码
    def decode(self,ctc_logits:torch.Tensor,vid_lgt:torch.Tensor,batch_first:bool=False,is_probability_distribution:bool=False):
        r"""
        ctc_logits: 模型的输出张量，一般是没有经过softmax的
        vid_lgt: 每一个样本的有效帧长度
        batch_first: batch是否位于张量的首位
        is_probability_distribution: 是否已经经过概率分布处理
        """
        if not batch_first:
            ctc_logits=ctc_logits.permute((1,0,2))
        if self.search_mode=="beam":
            return self.beam_search(ctc_logits,vid_lgt,is_probability_distribution)
        elif self.search_mode=="max":
            return self.max_search(ctc_logits,vid_lgt)
        else:
            # 没有找到这种解码方式
            logger.error(f"This decoding method was not found")
            raise ValueError(f"This decoding method was not found")
    # beam search 解码
    def beam_search(self,ctc_logits:torch.Tensor,vid_lgt:torch.Tensor,is_probability_distribution:bool=False):
        if not is_probability_distribution:
            ctc_logits=F.softmax(ctc_logits,dim=-1)
        pass
    # max search 解码（贪心）
    def max_search(self,ctc_logits:torch.Tensor,vid_lgt:torch.Tensor):
        pass
if __name__=="__main__":
    gloss_dict={
        "<blank>": 0,
        "我": 1,
        "是": 2,
        "学": 3,
        "生": 4,
        "老": 5,
        "师": 6
    }
    decode=Decode(gloss_dict,7,"beam")
    print(decode.decoder)