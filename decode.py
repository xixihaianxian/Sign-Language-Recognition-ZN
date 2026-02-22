import torch
from itertools import groupby
from torch.nn import functional as F
from torch import nn
from collections import defaultdict
from typing import Dict, List
from loguru import logger
from pyctcdecode import build_ctcdecoder
import warnings
warnings.filterwarnings("ignore", message="kenlm python bindings are not installed.*")

# 负责把模型输出(CTC logits)转化为可读的序列
class Decode:
    def __init__(
        self,
        gloss_dict: Dict[str, int],
        num_classes: int,
        search_mode: str,
        blank_id: int = 0,
    ):
        r"""
        gloss_dict: gloss到标签的字典
        num_classes：类别总数
        search_mode：解码方式（beam / max）
        blank_id：空白符的id
        """
        self.gloss_dict = gloss_dict
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        # gloss <-> id 映射
        self.gloss2id = defaultdict(int)
        for gloss, idx in gloss_dict.items():
            if idx == blank_id:
                continue
            self.gloss2id[gloss] = idx

        self.id2gloss = {idx: gloss for gloss, idx in self.gloss2id.items()}

        self.vocab = [""] + [
            chr(2000 + i) for i in range(1, self.num_classes)
        ]
        assert len(self.vocab) == self.num_classes

        self.decoder = build_ctcdecoder(
            labels=self.vocab,
            # kenlm_model_path=None  # 如需语言模型可在此添加
        )

    def decode(
        self,
        ctc_logits: torch.Tensor,
        vid_lgt: torch.Tensor,
        batch_first: bool = False,
        is_probability_distribution: bool = False,
    ):
        r"""
        ctc_logits: (B,T,V) 或 (T,B,V)
        vid_lgt: 每个样本的有效帧长度
        """
        if not batch_first:
            ctc_logits = ctc_logits.permute(1, 0, 2)

        if self.search_mode == "beam":
            return self.beam_search(ctc_logits, vid_lgt, is_probability_distribution)
        elif self.search_mode == "max":
            return self.max_search(ctc_logits, vid_lgt)
        else:
            logger.error("This decoding method was not found")
            raise ValueError("This decoding method was not found")

    def beam_search(self, ctc_logits, vid_lgt, is_probability_distribution=False):
        decoded_batch = list()
        token_ids_batch = list()

        if not is_probability_distribution:
            ctc_logits = F.log_softmax(ctc_logits, dim=-1)
        if ctc_logits.is_cuda:
            ctc_logits = ctc_logits.cpu()
        if vid_lgt.is_cuda:
            vid_lgt = vid_lgt.cpu()
        for b in range(ctc_logits.size(0)):
            T = int(vid_lgt[b])
            log_probs = ctc_logits[b][:T]  # (T, V)
            decoded_str = self.decoder.decode(log_probs.numpy())
            # char → token id
            token_ids = [
                self.vocab.index(ch)
                for ch in decoded_str
                if ch in self.vocab and self.vocab.index(ch) != self.blank_id
            ]
            # CTC collapse
            token_ids = [k for k, _ in groupby(token_ids)]

            if len(token_ids) > 0:
                decoded_batch.append([
                    (self.id2gloss[int(t)], i)
                    for i, t in enumerate(token_ids)
                ])
                token_ids_batch.append(torch.tensor(token_ids))
            else:
                decoded_batch.append([("EMPTY", 0)])
                token_ids_batch.append(torch.tensor([self.blank_id]))
        return decoded_batch, token_ids_batch

    def beam_search_c(self,ctc_logits: torch.Tensor,vid_lgt: torch.Tensor,is_probability_distribution: bool = False):
        decoded_batch = []
        results = []
        if not is_probability_distribution:
            ctc_logits = F.log_softmax(ctc_logits, dim=-1)
        else:
            ctc_logits = ctc_logits
        if ctc_logits.is_cuda:
            ctc_logits = ctc_logits.cpu()
        if vid_lgt.is_cuda:
            vid_lgt = vid_lgt.cpu()
        for b in range(ctc_logits.size(0)):
            T = int(vid_lgt[b])
            log_probs = ctc_logits[b][:T]
            decoded_str = self.decoder.decode(log_probs.numpy())
            token_ids = [
                self.vocab.index(ch)
                for ch in decoded_str
                if ch in self.vocab and self.vocab.index(ch) != self.blank_id
            ]
            token_ids = [k for k, _ in groupby(token_ids)]
            if len(token_ids) > 0:
                decoded_batch.append(
                    [(self.id2gloss[int(t)], i) for i, t in enumerate(token_ids)]
                )
                results.append(torch.tensor(token_ids))
            else:
                try:
                    decoded_batch.append(decoded_batch[-1])
                    results.append(results[-1])
                except Exception:
                    logger.warning(f"decoded batch len is {len(decoded_batch)}")
                    decoded_batch.append([("EMPTY", 0)])
                    results.append(torch.tensor([0]))
        return decoded_batch, results

    def max_search(self, ctc_logits: torch.Tensor, vid_lgt: torch.Tensor):
        decoded_batch = []
        goal_index = torch.argmax(ctc_logits, dim=2)
        for batch_index in range(len(ctc_logits)):
            group_result = [
                item[0]
                for item in groupby(goal_index[batch_index][: vid_lgt[batch_index]])
            ]
            filtered = [item for item in group_result if item != self.blank_id]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [item[0] for item in groupby(max_result)]
            else:
                max_result = filtered
            decoded_batch.append(
                [(self.id2gloss.get(int(item)), index) for index, item in enumerate(max_result)]
            )
        return decoded_batch

if __name__ == "__main__":
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
    # log_soft_max=nn.LogSoftmax(dim=0)
    # ctc_logits=log_soft_max(ctc_logits)
    decode=Decode(gloss_dict,len(gloss_dict),"beam")
    beam_search=decode.beam_search_c(ctc_logits,vid_lgt)
    max_result=decode.max_search(ctc_logits, vid_lgt)
    data_ctc=beam_search[1]
    print(data_ctc)