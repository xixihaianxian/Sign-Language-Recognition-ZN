import torch
from torch import nn
from torch.nn import functional as F

# 设置循环层封装器
class BiLSTMLayer(nn.Module):
    r"""
    Args
    input_size: 输入特征维度
    hidden_size: LSTM隐藏层维度
    num_layers: LSTM堆叠层数
    dropout: 层间dropout
    bidirectional: 是否双向
    rnn_type: rnn 类型
    debug: 调试开关
    """
    def __init__(self,embedding_size,hidden_size:int=512,num_layers=1,dropout=0,bidirectional:bool=True,rnn_type:str="LSTM",debug:bool=False):
        super().__init__()
        self.embedding_size=embedding_size
        self.num_layers=num_layers
        self.dropout=dropout
        self.bidirectional=bidirectional
        self.rnn_type=rnn_type
        self.debug=debug
        self.num_directions=2 if bidirectional else 1
        self.hidden_size = int(hidden_size/self.num_directions)
        self.rnn=getattr(nn,self.rnn_type)(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )
    def forward(self,src_features:torch.Tensor,src_lens:torch.Tensor,hidden:torch.Tensor=None):
        r"""
        Args
        src_features:（时间步长，批量大小，特征维度）
        src_lens:（每个序列的实际长度）
        hidden:（初始隐藏状态）(num_layers,batch_size,hidden_size)，在LSTM的情况下hidden应该是一个tuple，同时
        长度为2，每一个元素的形状都应该是（num_layers，batch_size，hidden_size）
        """
        packed_embedding=nn.utils.rnn.pack_padded_sequence(src_features,src_lens)
        # 防止提前就设置好hidden，但是hidden不符合LSTM需求的情况
        if hidden is not None and self.rnn_type=="LSTM" and not isinstance(hidden,tuple):
            split=int(hidden.size(0)/2)
            hidden=hidden[:split]
            cell=hidden[split:]
            hidden=(hidden,cell)
        # 将参数加入到模型
        packed_outputs,hidden=self.rnn(packed_embedding,hidden)
        # 经过模型训练之后hidden的形状为(num_layers*bidirectional,batch_size,hidden_size),
        # cell_hidden的shape同理
        rnn_outputs,_=nn.utils.rnn.pad_packed_sequence(packed_outputs)
        if self.bidirectional:
            hidden=self.cat_directions(hidden)
        if isinstance(hidden,tuple):
            hidden=torch.cat(hidden,0)
        return {
            "predictions":rnn_outputs,
            "hidden":hidden,
        }
    # 改变hidden的形状，将(num_layers*bidirectional,batch_size,hidden_size)转化为
    # (num_layers,batch_size,hidden_size*bidirectional)
    def cat_directions(self,hidden:torch.Tensor):
        r"""
        hidden经过循环神经网络计算得到的隐藏状态，这里需要对这个隐藏状态进行形状上的处理
        """
        def _cat(h:torch.Tensor):
            return torch.cat([h[0:h.size(0):2],h[1:h.size(0):2]],dim=2)
        if isinstance(hidden,tuple):
            return tuple([_cat(h) for h in hidden])
        else:
            return _cat(hidden)
if __name__=="__main__":
    # batch_size = 4
    # seq_len = 10
    # embedding_size = 128
    # hidden_size = 256
    # src_features = torch.randn(seq_len, batch_size, embedding_size)
    # src_lens = torch.tensor([10, 8, 6, 4])
    # init_hidden = None
    # bilstm=BiLSTMLayer(embedding_size, hidden_size)
    # print(bilstm(src_features,src_lens,init_hidden)["hidden"].shape)
    pass