import imageio
import os
import cv2
from tqdm import tqdm
import random
import numpy as np
from loguru import logger
import argparse

# 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"]=str(seed) # 固定 Python 的哈希随机种子
    random.seed(seed)
# 数据预处理
def data_preprocessing(origin_data_dir:str,save_dir:str):
    r"""
    Parameters
    ----------
    origin_data_dir: data origin path
    save_dir: save data path
    """
    # 创建存放数据的目录
    os.path.exists(save_dir) or os.makedirs(save_dir)
    logger.info(f"The {save_dir} directory has been created!")
    # 获取origin_data_dir里的目录(test,dev,train)
    file_types=sorted(os.listdir(origin_data_dir))
    frame_list=list() # 存放总帧数
    fps_list=list()# 存放fps，帧率（每秒里面有多少帧）
    video_time_list=list()# 存放视频时长，单位s
    resolution_list=list()# 存放分辨率
    # 对每一个类型的目录进行处理
    for file_type in file_types:
        # 获取某个类型的目录地址
        type_dir=os.path.join(origin_data_dir,file_type)
        # 单个类型的保存地址
        type_save_dir=os.path.join(save_dir,file_type)
        # 翻译者目录
        translators=sorted(os.listdir(type_dir))
        # 遍历每一个翻译者
        for translator in translators:
            translator_dir=os.path.join(type_dir,translator)
            translator_save_dir=os.path.join(type_save_dir,translator)
            # 视频列表
            video_list=sorted(os.listdir(translator_dir))
            logger.info(f"Currently processing all {file_type}->{translator} videos!")
            # 遍历每一个视频
            for video in tqdm(video_list):
                video_path=os.path.join(translator_dir,video)
                name, ext=os.path.splitext(video)
                # 构建存放视频每一帧的目录
                image_save_dir=os.path.join(translator_save_dir,name)
                os.path.exists(image_save_dir) or os.makedirs(image_save_dir)
                # 获取视频信息
                vid=imageio.get_reader(video_path)
                nframe=vid.count_frames() # 总帧
                fps=vid.get_meta_data().get("fps") # fps(每秒多少帧)
                video_time=vid.get_meta_data().get("duration") # 视频总时长
                resolution=vid.get_meta_data().get("size") # 分辨率（width，height）
                # 保存信息
                frame_list.append(nframe)
                fps_list.append(fps)
                video_time_list.append(video_time)
                resolution_list.append(resolution)
                # 遍历每一帧
                for frame_number in range(nframe):
                    try:
                        frame_image=vid.get_data(frame_number) # frame_image 某一帧的图片
                        frame_image=cv2.cvtColor(frame_image,code=cv2.COLOR_BGR2RGB)
                        # resize更改分辨率
                        frame_image=cv2.resize(frame_image,dsize=(255,255))
                        image_name=str(frame_number)
                        for i in range(5-len(image_name)):
                            image_name="0"+image_name
                        # 存放图片的路径
                        image_path=os.path.join(image_save_dir,f"{image_name}.jpg")
                        cv2.imencode(ext=".jpg",img=frame_image)[1].tofile(image_path)
                    except Exception as error:
                        logger.error(f"Find error, please check {nframe}, {image_path}!")
                        raise Exception(f"请检查总帧：{nframe},图片路径：{image_path}")
                # 关闭vid
                vid.close()
    # 数据分析
    max_nframe=max(frame_list) # 最长总帧
    min_nframe=min(frame_list) # 最短总帧
    max_video_time=max(video_time_list) # 最时长
    min_video_time=min(video_time_list) # 最短时长
    # 日志输入（可忽略）
    logger.info(f"max nframe is {max_nframe}, min nframe is {min_nframe}, max video time is {max_video_time}, min video time is {min_video_time}.")
    logger.info(f"Video data processed successfully!")
# 设置指令
def main():
    r"""
    构建指令
    """
    parser=argparse.ArgumentParser(description="Data preprocessing")
    # 数据原始位置
    parser.add_argument("--origin","-o",type=str,help="data origin path",default="/mnt/e/Sign-Language-Recognition-ZN/CE-CSL/video",dest="origin_data_path")
    # 存放预处理后数据的位置
    parser.add_argument("--save","-s",type=str,help="data save path",default="/mnt/e/Sign-Language-Recognition-ZN/data/video",dest="save_path")
    # 获取所有的参数
    args=parser.parse_args()
    # 构建逻辑
    data_preprocessing(args.origin_data_path, args.save_path)

if __name__=="__main__":
    # origin_data_path="/usr/Sign-Language-Recognition/CE-CSL/video"
    # save_path="/usr/Sign-Language-Recognition/data/video"
    # 执行预处理数据的操作
    # data_preprocessing(origin_data_path,save_path)
    main()