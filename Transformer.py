import torch
from torch import nn
import numpy as np
from einops import rearrange

# 根据序列长度生padding mask，为避免运算时模型注意到padding
def key_padding_mask(sequence_len):
    mask=torch.zeros(size=(len(sequence_len),max(sequence_len)),dtype=torch.float32)
    for index,length in enumerate(sequence_len):
        mask[index,length:]=True
    return mask

# 缩放点积注意力机制
class ScaledDotProductAttention(nn.Module):
    def __init__(self,dropout):
        super().__init__()
        self.dropout=nn.Dropout(p=dropout)
    def forward(self,q,k,v,mask,relative_position_encoding_q,relative_position_encoding_v):
        r"""
        params:
            q:query(查询参数)
            k:key(键向量)
            v:value(词向量)
            mask:mask(掩码)，避免模型关注padding
        """
        pass