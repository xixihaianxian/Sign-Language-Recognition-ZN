import torch
from torch import nn
import numpy as np
from einops import rearrange
import math
from typing import Dict
from loguru import logger

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
    def forward(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,
                mask:torch.Tensor=None,relative_position_encoding_q:torch.Tensor=None,
                relative_position_encoding_v:torch.Tensor=None)->Dict[str,torch.Tensor]:
        r"""
        params:
            q:query(查询参数)(*, query_len, dim)
            k:key(键向量)(*, key_len, dim)
            v:value(词向量)(*, key_len, dim)
            mask:mask(掩码)，避免模型关注padding(*, query_len, key_len)
            relative_position_encoding_q:相对Q侧的位置编码(query_len, key_len, dim)
            relative_position_encoding_v:相对V侧的位置编码(query_len, key_len, dim)
        """
        # 获取q的维度
        dim=q.size(-1)
        q/=math.pow(dim,0.5)
        energy=torch.matmul(q,k.transpose(-2,-1))
        if relative_position_encoding_q is not None:
            energy+=torch.einsum("...qd,qkd->...qk",q,relative_position_encoding_q)
        if mask is not None:
            energy=energy.masked_fill(mask,value=float('-inf'))
        alignment=torch.softmax(energy,dim=-1)
        context=torch.matmul(self.dropout(alignment),v)
        if relative_position_encoding_v is not None:
            context+=torch.einsum("...qk,qkd->...qd",alignment,relative_position_encoding_v)
        return {
            "context":context,
            "alignment":alignment,
        }

# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self,dim,head_number,dropout,relative_position_encoding_k):
        super().__init__()
        if dim%head_number!=0:
            logger.error(f"dim should be a multiple of head_number!")
            raise ValueError("dim should be a multiple of heads_number!")
        self.dim=dim # 维度
        self.n_head=head_number # 头数
        self.relative_position_encoding_k=relative_position_encoding_k
        self.w_q=nn.Linear(in_features=dim,out_features=dim)
        self.w_k=nn.Linear(in_features=dim,out_features=dim)
        self.w_v=nn.Linear(in_features=dim,out_features=dim)
        if self.relative_position_encoding_k>0:
            self.relative_position_encoding_w=nn.Embedding(num_embeddings=relative_position_encoding_k*2+1,embedding_dim=2*dim//self.n_head)
        self.scaled_attention=ScaledDotProductAttention(dropout)
        self.fc=nn.Linear(in_features=dim,out_features=dim)
    def forward(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask=None):
        r"""
        params:
            q: query (batch, query_len, dim)
            k: key (batch, key_len, dim)
            v: value (batch, key_len, dim)
            mask: mask(batch, query_len, key_len)
        """
        # 获取batch_size，query_len，key_len等参数
        batch_size=q.size(0)
        query_len=q.size(1)
        key_len=k.size(1)
        q=self.w_q(q)
        k=self.w_k(k)
        v=self.w_v(v)
        # 改变q,k,v的形状，如q(batch_size,query_len,dim)->q(batch_size,n_head,query_len,dim//n_head)
        split_heads=lambda x:rearrange(x,"b t (h d) -> b h t d",h=self.n_head)
        q,k,v=map(split_heads,(q,k,v))
        # 在mask的第二个维度上增加一个大小为1的维度
        if mask is not None:
            mask=mask.unsqueeze(1)
        if self.relative_position_encoding_k>0:
            distance=self.relative_distance(length=max(query_len,key_len),k=self.relative_position_encoding_k)
            distance=distance[:query_len,:key_len].to(device=q.device)
            relative_position_encoding_q,relative_position_encoding_v=self.relative_position_encoding_w(distance).chunk(2,dim=-1)
            context, alignment=self.scaled_attention(q,k,v,mask,relative_position_encoding_q,relative_position_encoding_v)
        else:
            context, alignment=self.scaled_attention(q,k,v,mask)
        # swap len and head back
        context=rearrange(context,pattern="b h t d -> b t (h d)")
        context=self.fc(context)
        return {
            "context":context,
            "alignment":alignment,
        }
    @staticmethod
    def relative_distance(length,k):
        indices=torch.arange(length)
        indices=indices.unsqueeze(1).expand(size=(-1,length))
        distance=indices-indices.transpose(0,1)
        distance=distance.clamp(min=-k,max=k)+k
        return distance
if __name__=="__main__":
    pass