import numpy as np
from typing import List,AnyStr

WER_COST_DEL = 1 # 删除的代价
WER_COST_INS = 1 # 插入的代价
WER_COST_SUB = 1 # 替换的代价

# 具体代码可以参考 https://github.com/zszyellow/WER-in-python/blob/master/wer.py

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
def edit_distance(reference:List[str],hypothesis:List[str]):
    # 初始化距离矩阵，加1是为了考虑空格
    distance=np.zeros(shape=((len(reference)+1)*(len(hypothesis)+1)),dtype=np.int32).reshape((len(reference)+1,len(hypothesis)+1))
    for row in range(len(reference)+1):
        for column in range(len(hypothesis)+1):
            if row==0:
                distance[0,column]=column*WER_COST_INS # 删除多少次变成预测字符
            elif column==0:
                distance[row,0]=row*WER_COST_DEL # 删除多少次变成空字符
    for row in range(1,len(reference)+1):
        for column in range(1,len(hypothesis)+1):
            if reference[row-1]==hypothesis[column-1]:
                distance[row,column]=distance[row-1,column-1] #如果想等，操作次数等于上一层的操作次数
            else:
                # 正常的编辑距离矩阵是从参考的角度来说的，所以insert是加上的是insert cost，但是如果你从预测的角度来说的话
                # 就需要加上delete cost，但是这里我们不做修改，因为答案是一样的
                insert=distance[row,column-1]+WER_COST_INS # 不相同时，增添后需要的执行次数
                delete=distance[row-1,column]+WER_COST_DEL # 不相同时，删除后需要的执行次数
                substitute=distance[row-1,column-1]+WER_COST_SUB # 不相同时，替换后需要的执行次数
                distance[row,column]=min(insert,delete,substitute)
    return distance
# 根据编辑距离矩阵回溯出参考序列与预测序列的词级对齐情况，我们的目标是将预测句子序列转化为参考的序列
# 所以与要站在预测的角度来修改
def get_alignment(reference:List[str],hypothesis:List[str],distance:np.ndarray):
    r"""
    reference: 参考句子的词序列
    hypothesis：预测句子的词序列
    distance: 编辑距离矩阵
    """
    reference_len=len(reference)
    hypothesis_len=len(hypothesis)
    max_len=reference_len+hypothesis_len # 计算出安全上限，避免死循环
    # 存储操作标记
    align_list=list()
    # 保存参考句对齐字符串
    align_reference=str()
    # 保存预测句对齐字符串
    align_hypothesis=str()
    # 保存操作标记字符串
    align_ment=str()
    while True:
        if (reference_len<=0 and hypothesis_len<=0) or (len(align_list)>max_len):
            break
        # 如果字符串相等
        elif reference_len>=1 and hypothesis_len>=1 and distance[reference_len,hypothesis_len]==distance[reference_len-1,hypothesis_len-1] and reference[reference_len-1]==hypothesis[hypothesis_len-1]:
            align_hypothesis=" "+hypothesis[hypothesis_len-1]+align_hypothesis
            align_reference=" "+reference[reference_len-1]+align_reference
            align_ment=" "*(len(reference[reference_len-1])+1)+align_ment
            align_list.append("C") # 相等时的标志
            # 更新 reference_len，和hypothesis_len
            reference_len=max(reference_len-1,0)
            hypothesis_len=max(hypothesis_len-1,0)
        # 如果是删除的情况，注意是在预测是的角度看来处理的
        elif hypothesis_len>=1 and distance[reference_len,hypothesis_len]==distance[reference_len,hypothesis_len-1]+WER_COST_DEL:
            align_reference=" "+"*"*len(hypothesis[hypothesis_len-1])+align_reference
            align_hypothesis=" "+hypothesis[hypothesis_len-1]+align_hypothesis
            align_ment=" "+"D"+" "*(len(hypothesis[hypothesis_len-1])-1)+align_ment
            align_list.append("D") # 删除时的标志
            reference_len=max(reference_len,0)
            hypothesis_len=max(hypothesis_len-1,0)
        # 添加的情况
        elif reference_len>=1 and distance[reference_len,hypothesis_len]==distance[reference_len-1,hypothesis_len]+WER_COST_INS:
            align_reference=" "+reference[reference_len-1]+align_reference
            align_hypothesis+" "+"*"*len(reference[reference_len-1])+align_hypothesis
            align_ment=" "+"I"+" "*(len(reference[reference_len-1])-1)+align_ment
            align_list.append("I") # 添加标志
            reference_len=max(reference_len-1,0)
            hypothesis_len=max(hypothesis_len,0)
        # 替换的情况
        elif reference_len>=1 and hypothesis_len>=1 and distance[reference_len,hypothesis_len]==distance[reference_len-1,hypothesis_len-1]+WER_COST_SUB:
            medium_len=max(len(reference[reference_len-1]),len(hypothesis[hypothesis_len-1]))
            align_reference=" "+reference[reference_len-1].ljust(medium_len)+align_reference
            align_hypothesis=" "+hypothesis[hypothesis_len-1].ljust(medium_len)+align_hypothesis
            align_ment=" "+"S"+" "*(medium_len-1)+align_ment
            align_list.append("S")
    # 删除多余的空格
    align_reference=align_reference[1:]
    align_hypothesis=align_hypothesis[1:]
    align_ment=align_ment[1:]
    # 将align_list倒置
    align_list=align_list[::-1]
    return (
        align_list,
        {
            "align_reference":align_reference,
            "align_hypothesis":align_hypothesis,
            "align_ment":align_ment,
        }
    )
if __name__=="__main__":
    # reference=["我","是","谁","她","不","人"]
    # hypothesis=["我","是","谁","她","不","是","人"]
    # print(edit_distance(reference,hypothesis))
    pass