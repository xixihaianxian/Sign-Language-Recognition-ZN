import torch
from itertools import groupby
import ctcdecode
from torch.nn import functional as F
from six.moves import xrange
from torch import nn
from collections import defaultdict
from typing import Dict,List
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
        ctc_logits: 模型的输出张量，一般是没有经过softmax的。形状(B,T,N)
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
        # 收集每个样本的decode结果
        decoded_batch=list()
        first_result=None
        # 对ctc_logits进行softmax
        if not is_probability_distribution:
            ctc_logits=F.softmax(ctc_logits,dim=-1)
        # 判断ctc_logits和vid_lgt所处设备
        if ctc_logits.is_cuda:
            ctc_logits=ctc_logits.to(device=torch.device("cpu"))
        if vid_lgt.is_cuda:
            vid_lgt=vid_lgt.to(device=torch.device("cpu"))
        # 调用解码器
        beam_result,beam_score,time_steps,out_seq_len=self.decoder.decode(ctc_logits,vid_lgt)
        # beam_result 每一个样本在beam search中得到的若干候选序列
        # beam_score 候选得分
        # time_steps 每一个token在时间维度上的起始帧
        # out_seq_len 每一个后选序列的有效长度
        for batch_index in range(len(ctc_logits)):
            first_result=beam_result[batch_index][0][:out_seq_len[batch_index][0]]
            if len(first_result)!=0:
                first_result=torch.stack([item[0] for item in groupby(first_result)])
            # 将索引转化为标签文字
            tmp=[(self.id2gloss[int(gloss_id)],index) for index,gloss_id in enumerate(first_result)]
            # 判断tmp是否为空
            if len(tmp)!=0:
                decoded_batch.append(tmp)
            else:
                try:
                    # 如果decode结果为空，将上一个样本的decode结果加入到容器中
                    decoded_batch.append(decoded_batch[-1])
                except Exception as error:
                    logger.warning(f"decoded batch len is {len(decoded_batch)}")
                    decoded_batch.append([("EMPTY", 0)])
        return decoded_batch,first_result
    # max search 解码（贪心）
    def max_search(self,ctc_logits:torch.Tensor,vid_lgt:torch.Tensor):
        decoded_batch=list()
        goal_index=torch.argmax(ctc_logits,dim=2)
        for batch_index in range(len(ctc_logits)):
            group_result=[item[0] for item in groupby(goal_index[batch_index][:vid_lgt[batch_index]])]
            filtered=[item for item in group_result if item != self.blank_id]
            if len(filtered)>0:
                max_result=torch.stack(filtered)
                # max_result=torch.tensor(filtered)
                max_result=[item[0] for item in groupby(max_result)]
            else:
                max_result=filtered
            decoded_batch.append([(self.id2gloss.get(int(item)),index) for index,item in enumerate(max_result)])
        return decoded_batch
# 计算前向对数似然
def ctc_loss(log_probs,targets,input_lengths,target_lengths,blank=0):
    r"""
    计算前向对数似然
    Args
        log_probs: 模型输出的对数概率
        targets: 拼接好的所有样本目标标签
        input_lengths: 每个样本的输入序列长度
        target_lengths: 每个样本的目标序列长度
        blank: 空白符索引
    """
    # 获取批量大小
    batch_size=len(target_lengths)
    n=0
    # 初始化
    llForward=0
    input_length_sum=0
    for batch_index in range(batch_size):
        # 目标序列长度
        seq_len=target_lengths[batch_index]
        extend_target_seq_len=2*seq_len+1
        # 输入序列长度
        input_seq_len=input_lengths[batch_index]
        # 建立alphas，前向概率表
        alphas=torch.zeros(size=(extend_target_seq_len,input_seq_len))
        # 建立betas，后向概率表
        betas=torch.zeros(size=(extend_target_seq_len,input_seq_len))
        # 对log_probs进行softmax
        log_probs=torch.softmax(log_probs,dim=-1)
        # 初始化
        alphas[0,0]=log_probs[0,batch_index,blank]
        alphas[1,0]=log_probs[0,batch_index,targets[n]]
        # 归一化系数
        normalization_constant=torch.sum(alphas[:,0])
        alphas[:,0]=alphas[:,0]/normalization_constant
        llForward=torch.log(normalization_constant)
        # 使用xrange减小内存的消耗
        for t in xrange(1,input_seq_len):
            # t时间步只有[start,end]这些状态是可能的
            start=max(0,extend_target_seq_len-2*(input_seq_len-t))
            end=min(2*t+2,extend_target_seq_len)
            for s in xrange(start,extend_target_seq_len):
                label=int((s-1)/2)
                # 如果s是偶数
                if s%2==0:
                    if s==0:
                        alphas[s,t]=alphas[s,t-1]*log_probs[t,batch_index,blank]
                    else:
                        alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * log_probs[t, batch_index, blank]
                # s=1且label和前一个label相同
                elif s==1 or targets[label]==targets[label-1]:
                    alphas[s,t]=(alphas[s,t-1]+alphas[s-1,t-1])*log_probs[t,batch_index,targets[label]]
                else:
                    alphas[s,t]=(alphas[s,t-1]+alphas[s-1,t-1]+alphas[s-2,t-1])*log_probs[t,batch_index,targets[label]]
            normalization_constant=torch.sum(alphas[start:end,t])
            alphas[start:end,t]=alphas[start:end,t]/normalization_constant
            llForward+=torch.log(normalization_constant)
        n+=target_lengths[batch_index]
        input_length_sum=torch.sum(input_lengths)
    return llForward/input_length_sum
if __name__=="__main__":
    gloss_dict = {"pad": 0, "hello": 1, "world": 2, "goodbye": 3}
    ctc_logits = torch.tensor([
        [  # 样本 1
            [2.0, 5.0, 0.5, 0.2],
            [1.0, 4.5, 0.3, 0.1],
            [0.2, 0.1, 6.0, 0.4],
            [0.3, 0.2, 5.5, 0.1],
            [0.1, 0.2, 0.3, 0.1],
            [3.0, 0.5, 0.2, 0.1],
            [2.5, 0.3, 0.2, 0.1],
            [1.0, 5.0, 0.3, 0.1],
            [0.1, 0.2, 6.0, 0.1],
            [0.3, 0.1, 5.5, 0.2],
        ],
        [  # 样本 2
            [3.0, 0.1, 0.2, 5.0],
            [2.5, 0.2, 0.3, 4.5],
            [0.1, 0.2, 6.0, 0.2],
            [0.3, 0.1, 5.5, 0.3],
            [1.0, 5.0, 0.3, 0.1],
            [0.1, 0.3, 0.2, 0.2],
            [2.0, 4.5, 0.3, 0.1],
            [0.1, 0.2, 6.0, 0.2],
            [0.3, 0.1, 5.5, 0.3],
            [0.1, 0.2, 0.3, 0.1],
        ]
    ])

    vid_lgt = torch.IntTensor([10, 10])
    decode=Decode(gloss_dict,len(gloss_dict),"beam")
    beam_search=decode.beam_search(ctc_logits,vid_lgt)
    max_result=decode.max_search(ctc_logits, vid_lgt)
    print(beam_search)
    print(max_result)
    # batch_size = 2
    # T_max = 5
    # vocab_size = 4
    # blank = 0
    # log_probs = torch.tensor([
    #     [
    #         [0.1, 0.6, 0.2, 0.1],
    #         [0.3, 0.3, 0.2, 0.2],
    #     ],
    #     [
    #         [0.7, 0.1, 0.1, 0.1],
    #         [0.25, 0.25, 0.25, 0.25],
    #     ],
    #     [
    #         [0.1, 0.2, 0.6, 0.1],
    #         [0.4, 0.2, 0.2, 0.2],
    #     ],
    #     [
    #         [0.25, 0.25, 0.25, 0.25],
    #         [0.1, 0.6, 0.2, 0.1],
    #     ],
    #     [
    #         [0.4, 0.3, 0.2, 0.1],
    #         [0.5, 0.1, 0.3, 0.1],
    #     ]
    # ], dtype=torch.float32)
    # targets = torch.tensor([1, 2, 1, 1, 2], dtype=torch.long)
    # input_lengths = torch.tensor([5, 5], dtype=torch.long)
    # target_lengths = torch.tensor([2, 3], dtype=torch.long)
    # print(ctc_loss(log_probs, targets, input_lengths, target_lengths))