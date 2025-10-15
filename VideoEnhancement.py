import torch
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from typing import List,Any
from loguru import logger

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