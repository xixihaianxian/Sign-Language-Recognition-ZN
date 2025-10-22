# author:YuHongfeng
# time:2025/10/20 11:07
# mood:😄
import torch
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from typing import List,Any
from loguru import logger
import numpy as np
import copy
import numbers
from PIL import Image
import random
import scipy
import matplotlib.pyplot as plt
import cv2

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
# TODO 可能有bug注意使用
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
        # logger.info(f"delete index is {delete_label_index}")
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
        insert_snippet_size=boundary_info[insert_label_position+1]-boundary_info[insert_label_position]
        if snippet_index!=0:
            boundary_info=boundary_info[:snippet_index]+[boundary_info[snippet_index-1]+insert_snippet_size]+[snippet+insert_snippet_size for snippet in boundary_info[snippet_index:]]
        else:
            boundary_info=[boundary_info[0]]+[boundary_info[0]+insert_snippet_size]+[snippet+insert_snippet_size for snippet in boundary_info[1:]]
        return video_frame_index,label,boundary_info
    # 替换操作
    @staticmethod
    def substitution(video_frame_index:List[int],label:List[int],boundary_info:List[int]):
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
# 视频随机裁剪
class RandomCrop(object):
    r"""
    使用前请保证video_sequence的数据类型是数组或者是PIL.Image.Image对象，不然会报错
    size: size is sequence or int. Notices: size shape is (height,width)
    """
    def __init__(self,size:Any):
        # size代表对每一帧裁剪之后输出的单帧的大小
        super().__init__()
        if isinstance(size,numbers.Number): # 如果size是数字的情况下的判断
            if size<=0:
                logger.error(f"If the input is a number, it must be greater than or equal to zero.")
                raise ValueError(f"If the input is a number, it must be greater than or equal to zero.")
            self.size=(size,size)
        else: # 如果size不是数字的情况下的判断
            if len(size)!=2:
                logger.error(f"If it is a set, the length can only be two.")
                raise ValueError(f"If it is a set, the length can only be two.")
            self.size=size
    def __call__(self,video_sequence):
        crop_height,crop_width=self.size
        # 随机抽取一帧
        random_index=np.random.choice(len(video_sequence))
        video_frame=video_sequence[random_index]
        # 获取原本帧的高宽
        if isinstance(video_frame,np.ndarray):
            image_height,image_width,image_channel=video_frame.shape
        elif isinstance(video_frame,Image.Image):
            image_width,image_height=video_frame.size
            # 将PIL.Image.Image类型转化为numpy.ndarray
            video_sequence = [np.array(frame.convert("RGB")) for frame in video_sequence]
        else:
            logger.error(f"Expected input is a numpy.ndarray or PIL.Image.Image，but input is {type(video_frame)}")
            raise ValueError(f"Expected input is a numpy.ndarray or PIL.Image.Image")
        # 随机裁剪
        if crop_height>image_height: # crop_height大于image_image是需要填充
            # 计算出需要填充的高
            pad_height=crop_height-image_height
            video_sequence=[np.pad(frame,pad_width=((pad_height//2,pad_height-pad_height//2),(0,0),(0,0)),mode="constant",constant_values=0) for frame in video_sequence]
            height_start=0
        else:
            height_start=np.random.randint(low=0,high=image_height-crop_height)
        if crop_width>image_width:
            # 计算出需要填充的宽
            pad_width=crop_width-image_width
            video_sequence=[np.pad(frame,pad_width=((0,0),(pad_width//2,pad_width-pad_width//2),(0,0)),mode="constant",constant_values=0) for frame in video_sequence]
            width_start=0
        else:
            width_start=np.random.randint(low=0,high=image_width-crop_width)
        video_sequence=[frame[height_start:height_start+crop_height,width_start:width_start+crop_width,:] for frame in video_sequence]
        # 将列表转化为数组
        video_sequence=np.array(video_sequence)
        return video_sequence
# 中心裁剪
class CenterCrop(RandomCrop):
    def __init__(self,size):
        super().__init__(size)
    def __call__(self,video_sequence):
        # 随便抽取一帧
        choice_frame=np.random.choice(len(video_sequence))
        video_frame=video_sequence[choice_frame]
        # 获取原帧的数据
        if isinstance(video_frame,np.ndarray):
            image_height,image_width,channel=video_frame.shape
        elif isinstance(video_frame,Image.Image):
            image_width,image_height=video_frame.size
            # 将PIL.Image.Image转化为numpy数组
            video_sequence=[np.array(frame.convert("RGB")) for frame in video_sequence]
        else:
            logger.error(f"Only supports PIL.Image.Image and numpy.ndarray")
            raise TypeError(f"Only supports PIL.Image.Image and numpy.ndarray")
        new_height,new_width=self.size
        # 确定裁剪之后的宽高
        new_height=image_height if new_height>=image_height else new_height
        new_width=image_width if new_width>=image_width else new_width
        # 确定裁剪起点
        top=int(round((image_height-new_height)/2.))
        left=int(round((image_width-new_width)/2.))
        # 裁剪
        video_sequence=[frame[top:top+new_height,left:left+new_width,:] for frame in video_sequence]
        video_sequence=np.array(video_sequence)
        return video_sequence
# 随机反转
class RandomHorizontalFlip:
    def __init__(self,prob):
        self.prob=prob
    def __call__(self,video_sequence:np.ndarray)->np.ndarray:
        # 计算概率，此时需要确定video_sequence的形状为（batch_size，height，width，channel）
        if not isinstance(video_sequence,np.ndarray):
            video_sequence=np.array([np.array(frame) for frame in video_sequence])
        flag=random.random()
        # 水平翻转
        if flag<self.prob:
            video_sequence=np.flip(video_sequence,axis=2)
            video_sequence=np.ascontiguousarray(video_sequence)
        return video_sequence
# 随机旋转
class RandomRotation:
    def __init__(self,angle):
        if isinstance(angle,numbers.Number):
            if angle>0:
                self.angle=(-angle,angle)
            else:
                logger.warning(f"It is recommended to enter a number greater than 0 for the angle.")
                self.angle=(angle,-angle)
        else:
            # 如果angle是序列的那么angel的长度应该是2，当超出这个长度的时候就会报错
            if len(angle)!=2:
                logger.error(f"If angle is a sequence, its length should be 2.")
                raise Exception(f"If angle is a sequence, its length should be 2.")
            else:
                self.angle=angle
    def __call__(self,video_sequence=None):
        rotation_angle=np.random.uniform(low=self.angle[0],high=self.angle[1])
        # 数组旋转
        if isinstance(video_sequence[0],np.ndarray):
            video_sequence=[scipy.ndimage.rotate(input=frame,angle=rotation_angle,reshape=False,mode="constant",cval=0) for frame in video_sequence]
        # Image.Image对象旋转
        elif isinstance(video_sequence[0],Image.Image):
            video_sequence=[np.array(frame.rotate(angle=rotation_angle,expand=False,fillcolor=None).convert("RGB")) for frame in video_sequence]
        else:
            logger.error(f"video sequence only support numpy.ndarray or PIL.Image.Image!")
            raise TypeError(f"video sequence only support numpy.ndarray or PIL.Image.Image!")
        return np.array(video_sequence)
# 时间重采样
class TemporalRescale:
    def __init__(self,temp_scaling=0.2,frame_interval=1):
        r"""
        temp_scaling: 时间缩放比例，控制速度变化范围。比如0.2代表视频的长度在原来长度是80%和120%之间变化
        frame_interval=1：帧间隔，用于计算最大长度。 每个多少帧取一阵
        """
        self.min_len=32 # 最小帧（总帧数）
        self.max_len=int(np.ceil(230/frame_interval)) #最大帧（总帧数）
        self.L=1.0-temp_scaling # 最小缩放比例
        self.U=1.0+temp_scaling # 最大缩放比例
    def __call__(self,video_sequence):
        scale=4 # 对齐因子
        original_len=len(video_sequence) # 视频的总帧数
        # 计算修改之后的长度
        alter_len=int(original_len*(self.L+(self.U-self.L)*np.random.random()))
        if alter_len<self.min_len:
            alter_len=self.min_len
        if alter_len>self.max_len:
            alter_len=self.max_len
        if (alter_len-scale)%scale!=0:
            alter_len+=scale-(alter_len-scale)%scale
        if alter_len<=original_len:
            # 如果alter的长度小于original_len那就没有重复的再original video里面抽取帧
            index=sorted(random.sample(range(original_len),k=alter_len))
        else:
            # 如果alter_len大于original_len那就需要重复的再original video里面抽取帧
            index=sorted(random.choices(range(original_len),k=alter_len))
        alter_sequence=video_sequence[index]
        return alter_sequence
# 随机更改形状
class RandomResize:
    def __init__(self,rate,interpolation="bilinear"):
        r"""
        rate: 控制随机缩放范围
        inter: 差值方式，控制缩放之后图片的平滑程度
        """
        self.rate=rate
        self.interpolation=interpolation
    def __call__(self,video_sequence):
        # 缩放比例
        scale=random.uniform(a=1-self.rate,b=1+self.rate)
        image_height,image_width,channel=video_sequence[0].shape
        if isinstance(video_sequence[0],np.ndarray):
            alter_shape=(image_width*scale,image_height*scale)
            # opencv resize
            video_sequence=[cv2.resize(frame,dsize=alter_shape,interpolation=self.get_cv_interpolation(self.interpolation)) for frame in video_sequence]
        elif isinstance(video_sequence[0],Image.Image):
            image_width,image_height=video_sequence[0].size
            alter_shape=(image_width*scale,image_height*scale)
            # PIL resize
            video_sequence=[frame.resize(size=alter_shape,resample=self.get_PIL_interpolation(self.interpolation)) for frame in video_sequence]
    def get_PIL_interpolation(self,interpolation):
        if interpolation=="nearest":
            return cv2.INTER_NEAREST
        elif interpolation=="bilinear":
            return cv2.INTER_LINEAR
        elif interpolation=="bicubic":
            return cv2.INTER_CUBIC
        elif interpolation=="lanczos":
            return cv2.INTER_LANCZOS4
        elif interpolation=="area":
            return cv2.INTER_AREA
    def get_cv_interpolation(self,interpolation):
        if interpolation=="nearest":
            return Image.NEAREST
        elif interpolation=="lanczos":
            return Image.LANCZOS
        elif interpolation=="bilinear":
            return Image.BILINEAR
        elif interpolation=="bicubic":
            return Image.BICUBIC
        elif interpolation=="area":
            return Image.BOX
if __name__=="__main__":
    video=np.random.randn(5,255,255)