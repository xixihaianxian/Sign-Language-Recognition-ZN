import pickle
from typing import List
import config
import pandas as pd
import csv

# 删除非法字符
def remove_illegal_char(word:str)->str:
    # 定义栈，存放非法字符
    stack=list()
    # 存放结果
    result_chars=list()
    # 便利每一个字符
    for char in word:
        # 判断是不是开符号
        if char in config.OPEN_CHAR:
            stack.append(char) # 如果是开符号，加入栈
        elif char in config.CLOSE_CHAR_MAP.keys(): # 判断是不是闭符号
            # 判断栈最后一个符合是否和闭符号匹配
            if stack and stack[-1]==config.CLOSE_CHAR_MAP.get(char):
                stack.pop() # 匹配，删除stack里面最后一个符号
            else:
                pass
        # 将合法的符号加入result_chars
        else:
            if not stack:
                result_chars.append(char)
    # 返回结果
    return "".join(result_chars)
# 预处理words
def handle_words(words:List[str])->List[str]:
    r"""
    处理words里面的word
    """
    # 遍历每一个word
    for i in range(len(words)):
        word=words[i]
        # 删除word不必要的部分
        word=remove_illegal_char(word)
        # 删除不必要的数字
        if word[-1].isdigit():
            if not word[0].isdigit():
                word=word[:-1]
        # 统一符号
        if word[0]=="," or word[0]=="，": #TODO 如何使用分词的话这里需要进行相应的修改
            word="，"+word[1:]
        if word[0]=="?" or word[0]=="？":
            word="？"+word[1:]
        # if word=="," or word=="，":
        #     word="，"
        # if word=="?" or word=="？":
        #     word="？"
        # 去掉数字前导零
        if word.isdigit():
            word=str(int(word))
        words[i]=word
    return words
# 建word到id的映射
def word2id(train_label_path:str,valid_label_path:str,test_label_path:str,data_set_name:str):
    # 定义word list
    word_list = list()
    # label路径列表
    label_paths=[train_label_path, test_label_path, valid_label_path]
    # RWTH数据处理方式，可以用来微调中文模型
    if data_set_name=="RWTH":
        # 处理train label
        train_df=pd.read_csv(train_label_path,sep="|")
        train_annotations=train_df.loc[:,"annotation"]
        # 遍历每一句
        for annotation in train_annotations:
            words=annotation.split()
            word_list.extend(words)
        # 处理test label
        test_df=pd.read_csv(test_label_path,sep="|")
        test_annotations=test_df.loc[:,"annotation"]
        for annotation in test_annotations:
            words=annotation.split()
            word_list.extend(words)
        # 处理valid label
        valid_df=pd.read_csv(valid_label_path,sep="|")
        valid_annotations=valid_df.loc[:,"annotation"]
        for annotation in valid_annotations:
            words=annotation.split()
            word_list.extend(words)
    # 处理CE-CSL
    elif data_set_name=="CE-CSL":
        # 处理train data
        train_df=pd.read_csv(train_label_path,sep=",")
        train_gloss=train_df.loc[:,"Gloss"]
        for gloss in train_gloss:
            words=gloss.split("/")
            words=handle_words(words)
            word_list.extend(words)
        # 处理valid data
        valid_df=pd.read_csv(valid_label_path,sep=",")
        valid_gloss=valid_df[:,"Gloss"]
        for gloss in valid_gloss:
            words=gloss.split("/")
            words=handle_words(words)
            word_list.append(words)
        # 处理test data
        test_df=pd.read_csv(test_label_path,sep=",")
        test_gloss=test_df[:,"Gloss"]
        for gloss in test_gloss:
            words=gloss.split("/")
            words=handle_words(words)
            word_list.append(words)
    # 处理RWTH-T
    elif data_set_name == "RWTH-T":
        # 遍历所有label path
        for label_path in label_paths:
            # 简历上下文管理器，打开label path
            with open(label_path, 'r', encoding="utf-8") as file:
                reader = csv.reader(file)
                for index, row in enumerate(reader):
                    # 排除标签行
                    if index != 0:
                        row_str_list = row[0].split("|")
                        words =row_str_list[5].split()
                        word_list.extend(words)
    # 处理CSL-Daily
    elif data_set_name=="CSL-Daily":
        words_pkl="/mnt/e/Sign-Language-Recognition-ZN/labelData/CSL-Daily/csl2020ct_v2.pkl"
        with open(words_pkl,mode="rb") as pkl:
            data=pickle.load(pkl)
            word_list=handle_words(data["gloss_map"])

if __name__=="__main__":
    print(remove_illegal_char("a(b)c)"))