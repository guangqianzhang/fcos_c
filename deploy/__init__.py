from .Fcos3d import fcos3d
import yaml

from deploy.transforms_3d import RandomFlip3D
from deploy.formating import DefaultFormatBundle3D,Collect3D
from deploy.loadimage import LoadImageFromFileMono3D
from deploy.aug import MultiScaleFlipAug
__factory = {
    'fcos3d': fcos3d
}
print('hallo init')


def build_model(arch,config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return __factory[arch](config[arch])