import numpy as np
from typing import List,AnyStr

# 批量计算平均的WER、删除率、插入率、替换率。
def wer_list(references, hypotheses):
    r"""
    references: 参考文本的集合, 正确的句子/标签
    hypotheses: 预测文本的集合
    """
    # 初始化错误总数，删除总数，插入总数，替换总数，参考文所有词的总数
    total_error=total_del=total_ins=total_sub=total_ref_len=0
    # 真实的标签和预测是标签一一对应
    for reference, hypothesis in zip(references,hypotheses):
        pass
# 计算单个样本的WER，删除率，插入率，替换率
def wer_single(reference:AnyStr, hypothesis:AnyStr):
    r"""
    reference: 参考文本, 单个样本的参考文本
    hypothesis: 预测文本, 单个样本的预测文本
    """
    reference_list=reference.strip().split()
    hypothesis_list=hypothesis.strip().split()
    pass
# 计算预测句子和参考句子之间的编辑距离矩阵
def edit_distance(reference:List[str],hypotheses:List[str]):
    # 初始化距离矩阵
    distance=np.zeros(shape=((len(reference)+1)*(len(hypotheses)+1)),dtype=np.uint8).reshape(shape=(len(reference)+1,len(hypotheses)+1))
    for row in range(len(reference)+1):
        for column in range(len(hypotheses)+1):
            pass