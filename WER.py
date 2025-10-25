import numpy as np
from typing import List,AnyStr

WER_COST_DEL = 1 # 删除的代价
WER_COST_INS = 1 # 插入的代价
WER_COST_SUB = 1 # 替换的代价

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
# 计算预测句子和参考句子之间的编辑距离矩阵，动态规划
def edit_distance(reference:List[str],hypotheses:List[str]):
    # 初始化距离矩阵，加1是为了考虑空格
    distance=np.zeros(shape=((len(reference)+1)*(len(hypotheses)+1)),dtype=np.uint8).reshape(shape=(len(reference)+1,len(hypotheses)+1))
    for row in range(len(reference)+1):
        for column in range(len(hypotheses)+1):
            if row==0:
                distance[0][column]=column*WER_COST_DEL # 删除多少次变成空字符
            elif column==0:
                distance[row][0]=row*WER_COST_INS # 空字符增添次变成目标字符
    for row in range(1,len(reference)+1):
        for column in range(1,len(hypotheses)+1):
            if reference[row-1]==hypotheses[column-1]:
                distance[row][column]=distance[row-1][column-1] #如果想等，操作次数等于上一层的操作次数
            else:
                insert=distance[row][column-1]+WER_COST_INS # 不相同时，增添后需要的执行次数
                delete=distance[row-1][column]+WER_COST_DEL # 不相同时，删除后需要的执行次数
                substitute=distance[row-1][column-1]+WER_COST_SUB # 不相同时，替换后需要的执行次数
                distance[row][column]=min(insert,delete,substitute)
    return distance