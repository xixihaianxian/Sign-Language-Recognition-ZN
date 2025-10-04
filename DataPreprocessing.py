import imageio
import os
import cv2
from tqdm import tqdm
import random
import numpy as np

# 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"]=str(seed)
    random.seed(seed)
# 数据预处理
def data_preprocessing(origin_data_path,save_path):
    pass

if __name__=="__main__":
    pass