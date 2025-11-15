import torch
from torch import nn
import numpy as np
from einops import rearrange
import math
from typing import Dict
from loguru import logger

# 根据序列长度生padding mask，为避免运算时模型注意到padding
def key_padding_mask(sequence_len)->torch.Tensor:
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
    def __init__(self,dim,head_number,dropout,relative_position_encoding_k=0):
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
    def forward(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask=None)->Dict[str,torch.Tensor]:
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
            # 返回的是字典
            context, alignment=self.scaled_attention(q,k,v,mask,relative_position_encoding_q,relative_position_encoding_v).values()
        else:
            context, alignment=self.scaled_attention(q,k,v,mask).values()
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

# 位置逐元素反馈
class PositionWiseFeedForward(nn.Module):
    def __init__(self,dim,hidden,dropout:float):
        super().__init__()
        self.w1=nn.Linear(in_features=dim,out_features=hidden)
        self.w2=nn.Linear(in_features=hidden,out_features=dim)
        self.dropout=nn.Dropout(p=dropout)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        y=self.w1(x)
        y=self.relu(y)
        y=self.dropout(y)
        y=self.w2(y)
        return y

class PreNorm(nn.Module):
    def __init__(self,dim,model):
        super().__init__()
        self.norm=nn.LayerNorm(dim)
        self.model=model
    def forward(self,x):
        y=self.model(self.norm(x))
        return y

class Residual(nn.Sequential):
    def __init__(self,*layers):
        super().__init__(*layers)
    def forward(self,x):
        return super().forward(x)+x

# Transformer解码块
class Applier(nn.Module):
    def __init__(self,model, applier):
        super().__init__()
        self.model=model
        self.applier=applier
    def forward(self,x):
        return self.applier(self.model,x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self,dim,n_head,dropout,relative_position_encoding_k=0):
        super().__init__()
        multi_attention=MultiHeadAttention(dim=dim,head_number=n_head,dropout=dropout,
                                           relative_position_encoding_k=relative_position_encoding_k)
        ffn=PositionWiseFeedForward(dim=dim,hidden=4*dim,dropout=dropout)
        wrap=lambda m:Residual(PreNorm(dim=dim,model=m),nn.Dropout(p=dropout))
        self.attention=wrap(Applier(multi_attention,lambda model,x:model(x,x,x,self.x_mask).get("context")))
        self.ffn=wrap(ffn)
    def forward(self,x,x_mask):
        self.x_mask=x_mask # 延迟访问
        y=self.attention(x)
        del self.x_mask # 删除该属性
        y=self.ffn(y)
        return y

# Transformer解码
class TransformerEncoder(nn.Module):
    def __init__(self,dim,n_head,num_layers,dropout=0.1,relative_position_encoding_k=8):
        super().__init__()
        self.layers=nn.ModuleList()
        self.norm=nn.LayerNorm(dim)
        for _ in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    dim=dim,
                    n_head=n_head,
                    dropout=dropout,
                    relative_position_encoding_k=relative_position_encoding_k,
                )
            )
    def forward(self,x:torch.Tensor):
        x_length=list(map(len,x))
        x_mask=key_padding_mask(x_length).to(x.device)
        x_mask=x_mask.unsqueeze(dim=1)
        for layer in self.layers:
            x=layer(x,x_mask)
        y=self.norm(x)
        return y
if __name__=="__main__":
    x = torch.tensor([
        [  # batch 0 (实际长度 3，已 pad 到 5)
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # padding
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # padding
        ],
        [  # batch 1 (实际长度 5)
            [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
            [1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2],
            [3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8],
            [5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4],
            [6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0],
        ]
    ], dtype=torch.float32)  # shape (2, 5, 8)
    transformer = TransformerEncoder(dim=8, n_head=4, num_layers=5,dropout=0.5)
    result=transformer(x)
    print(result.shape)