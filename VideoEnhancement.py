# 今天心情😊
import torch
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from typing import List,Any
from loguru import logger
import numpy as np
import copy

# 构造transform组合组件，可以将多个transform组合起来使用
class Compose(object):
    def __init__(self,transforms:List[Any]):
        super().__init__()
        self.transforms=transforms
    def __call__(self,image):
        if not isinstance(transforms,list):
            logger.error(f"transforms must be a list")
            raise TypeError(f"transforms must be a list")
        else:
            for transform in self.transforms:
                image=transform(image)
        return image
# 对视频序列进行删除，插入，替换操作
class WERAugment(object):
    def __init__(self,boundary_path,k):
        super().__init__()
        self.boundary_path=boundary_path
        # 读取视频分段信息，同时转化为list
        self.boundary_dict=np.load(file=self.boundary_path,allow_pickle=True).item()
        # 最大操作次数
        self.k=k
    # 对视频序列进行相关的操作
    def __call__(self,video,label,file_info):
        r"""
        video 存放视频序列。可以是numpy数组，也可以是tensor张量，列表也可以（这些类型都可参与变换，但是需要注意的是根据模型需求来进行变换）
        label 包含了每小段视频的标签
        file_label 视频的唯一标识符，一般是视频的名称
        """
        # 视频的帧数量序列
        video_frame_index=np.arange(len(video)).tolist()
        # 判断file_info是否存在boundary_dict，存在说明没有这个视频的信息
        if file_info not in self.boundary_dict.keys():
            logger.error(f"video {file_info} not found")
            raise KeyError(f"video {file_info} not found")
        # 获取边界信息
        boundary_info=copy.deepcopy(self.boundary_dict.get(file_info))
        # 补全开头和结尾的边界
        boundary_info=[0]+boundary_info+[len(video)]
        # 确定修改次数，防止所有的帧都被删除
        k=np.random.randint(min(self.k,max(len(label)-1,1)))
        # 进行k轮d操作
        for n in range(k):
            video_frame_index,label,boundary_info=self.one_operation(video_frame_index,label,boundary_info)
        transform_video=[video[index] for index in video_frame_index]
        return transform_video,label
    # 定义好操作的方式
    def one_operation(self,*params):
        # 获取概率
        rate=np.random.random()
        if rate<0.3:
            return self.delete(*params)
        elif 0.3<=rate<0.7:
            return self.insert(*params)
        else:
            return self.substitution(*params)
    # 删除操作
    @staticmethod
    def delete(video_frame_index:List[int],label:List[int],boundary_info:List[int]):
        delete_label_index=np.random.randint(len(label))
        # 更新之后的video
        video_frame_index=video_frame_index[:boundary_info[delete_label_index]]+video_frame_index[boundary_info[delete_label_index+1]:]
        # 删除片段的大小
        delete_snippet_size=boundary_info[delete_label_index+1]-boundary_info[delete_label_index]
        # 更新之后发边界信息
        boundary_info=boundary_info[:delete_label_index]+[snippet-delete_snippet_size for snippet in boundary_info[delete_label_index+1:]]
        # 更新之后的label
        label.pop(delete_label_index)
        return video_frame_index,label,boundary_info
    # 插入操作
    @staticmethod
    def insert(video_frame_index:List[int],label:List[int],boundary_info:List[int]):
        # 选择复制哪个片段
        insert_label_position=np.random.randint(len(label))
        # 需要插入的位置(帧位置)
        insert_snippet=np.random.choice(boundary_info)
        # 插入片段的帧索引
        snippet_index=boundary_info.index(insert_snippet)
        video_frame_index=video_frame_index[:insert_snippet]+video_frame_index[boundary_info[insert_label_position]:boundary_info[insert_label_position+1]]+video_frame_index[insert_snippet:]
        label=label[:snippet_index]+[label[insert_label_position]]+label[snippet_index:]
        insert_snippet_size=boundary_info[snippet_index+1]-boundary_info[snippet_index]
        boundary_info=boundary_info[:snippet_index]+[boundary_info[snippet_index-1]+insert_snippet_size]+[snippet+insert_snippet_size for snippet in boundary_info[snippet_index:]]
        return video_frame_index,label,boundary_info
    # 替换操作
    @staticmethod
    def substitution(video_frame_index,label,boundary_info):
        # 使用哪个替换(索引)
        substitution_index=np.random.randint(len(label))
        # 需要替换的片段(索引)
        target_index=np.random.randint(len(label))
        video_frame_index=video_frame_index[:boundary_info[target_index]]+video_frame_index[boundary_info[substitution_index]:boundary_info[substitution_index+1]]+video_frame_index[boundary_info[target_index+1]:]
        # label=label[:target_index]+[label[substitution_index]]+label[target_index+1:]
        label[target_index]=label[substitution_index]
        substitution_snippet_size=boundary_info[substitution_index+1]-boundary_info[substitution_index]-(boundary_info[target_index+1]-boundary_info[target_index])
        boundary_info=boundary_info[:target_index+1]+[snippet+substitution_snippet_size for snippet in boundary_info[target_index+1:]]
        return video_frame_index,label,boundary_info
# 转化为Tensor张量
class ToTensor(object):
    def __init__(self):
        super().__init__()
    def __call__(self,data):
        if isinstance(data,list): # list转化为tensor
            data=torch.tensor(data,dtype=torch.float32)
        elif isinstance(data,torch.Tensor):# 转变为float32
            data=data.to(dtype=torch.float32)
        elif isinstance(data,np.ndarray):# 数组转化tensor
            data=torch.from_numpy(data).to(dtype=torch.float32)
        else:
            logger.error(f"It is an unsupported type")
            raise TypeError(f"It is an unsupported type")
        return data
# 随机裁剪
class RandomCrop(object):
    r"""
    size: size is sequence or int
    """
    def __init__(self):
        super().__init__()
if __name__=="__main__":
    pass