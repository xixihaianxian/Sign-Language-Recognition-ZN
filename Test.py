import numpy as np
import torch
import readconfig
from loguru import logger

def fetch_test_params():
    config_params=readconfig.read_config()
    # 数据路径
    test_data_path=config_params["testDataPath"]
    # 标签路径
    test_label_path=config_params["testLabelPath"]

def test():
    pass