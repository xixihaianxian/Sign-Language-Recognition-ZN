import torch
import configparser
import os
from loguru import logger

def read_config():
    # 默认参数
    config_params = {
        "trainDataPath":"/mnt/e/Sign-Language-Recognition-ZN/data/video/train",
        "validDataPath": "/mnt/e/Sign-Language-Recognition-ZN/data/video",
        "testDataPath": "/mnt/e/Sign-Language-Recognition-ZN/data/video/test",
        "trainLabelPath": "/mnt/e/Sign-Language-Recognition-ZN/CE-CSL/label/train.csv",
        "validLabelPath": "/mnt/e/Sign-Language-Recognition-ZN/CE-CSL/label/dev.csv",
        "testLabelPath": "/mnt/e/Sign-Language-Recognition-ZN/CE-CSL/label/test.csv",
        "bestModuleSavePath": "module/best.pth",
        "currentModuleSavePath": "module/current.pth",
        "device": "cuda", #或者  0:CPU  1:GPU
        "hiddenSize":512,
        "lr": 0.1,
        "batchSize": 1,
        "numWorkers": 2,
        "pinMemory": 1,
        "dataSetName": "CE-CSL",
    }
    # 获取自定义参数
    config_path="params.ini"
    if os.path.exists(config_path):
        cf=configparser.ConfigParser()
        cf.read(config_path)
        # 修改路径参数
        config_params["trainDataPath"]=cf.get("Path","trainDataPath")
        config_params["validDataPath"]=cf.get("Path","validDataPath")
        config_params["testDataPath"]=cf.get("Path","testDataPath")
        config_params["trainLabelPath"]=cf.get("Path","trainLabelPath")
        config_params["testLabelPath"]=cf.get("Path","testLabelPath")
        config_params["validLabelPath"]=cf.get("Path","validLabelPath")
        config_params["bestModuleSavePath"]=cf.get("Path","bestModuleSavePath")
        config_params["currentModuleSavePath"]=cf.get("Path","currentModuleSavePath")
        # 修改训练参数
        config_params["device"]=cf.get("Params","device")
        config_params["hiddenSize"]=cf.get("Params","hiddenSize")
        config_params["lr"]=cf.get("Params","lr")
        config_params["batchSize"]=cf.get("Params","batchSize")
        config_params["numWorkers"]=cf.get("Params","numWorkers")
        config_params["pinMemory"]=cf.get("Params","pinMemory")
        config_params["moduleChoice"] = cf.get("Params", "moduleChoice")
        config_params["dataSetName"]=cf.get("Params","dataSetName")
        # 判断是否存在GPU
        if torch.cuda.is_available():
            if config_params["device"]=="cuda":
                config_params["device"]=torch.device("cuda")
            else:
                config_params["device"]=torch.device("cpu")
        else:
            logger.warning(f"GPU does not exist!")
            config_params["device"]=torch.device("cpu")
    else:
        logger.warning(f"{config_params} is not exist!")
        logger.info(f"Use default params!")

    return config_params