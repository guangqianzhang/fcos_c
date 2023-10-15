import cv2
import numpy as np
from deploy.transforms_3d import RandomFlip3D
from deploy.formating import DefaultFormatBundle3D,Collect3D
from deploy.loadimage import LoadImageFromFileMono3D
from deploy.aug import MultiScaleFlipAug
from register import register


def preprocess_image(config):
    # 检查字典中是否存在 'type' 键，并获取对应的值
    if 'type' in config:
        preprocess_type = config['type']
    else:
        raise ValueError("Preprocessing type not specified in the configuration.")
    if preprocess_type=='LoadImageFromFileMono3D':
        return LoadImageFromFileMono3D()
    elif preprocess_type == 'Normalize':
        return Normalize(config)
    elif preprocess_type == 'Pad':
        return Pad(config)
    elif preprocess_type == 'RandomFlip3D':
        return RandomFlip3D(config)
    elif preprocess_type == 'DefaultFormatBundle3D':
        return DefaultFormatBundle3D(config)
    elif preprocess_type=='Collect3D':
        return Collect3D(config)
    elif preprocess_type=='MultiScaleFlipAug':
        return MultiScaleFlipAug()
