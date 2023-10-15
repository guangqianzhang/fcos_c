import os.path as osp

import cv2

from register import register
import numpy as np


class LoadImage:
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr',
                 ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        img = results['image']
        # TODO 如何判断设置img的channel
        if self.to_float32:
            img = img.astype(np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"channel_order='{self.channel_order}', "
                    )
        return repr_str

@register.register
class LoadImageFromFileMono3D(LoadImage):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        results['cam2img'] = results['img_info']['cam_intrinsic']
        return results


@register.register
class Normalize:
    def __init__(self):
        self.mean = [103.53, 116.28, 123.675]
        self.std = [1.0, 1.0, 1.0]
        self.to_rgb = False


    def __call__(self, reslut):
        return self.normalize_image(reslut)

    def normalize_image(self, reslut):
        # 转换图像数据类型为浮点型
        image = reslut['img'].astype(np.float32)
        # 归一化处理
        image = (image - self.mean) / self.std
        if self.to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        reslut['img']=image
        return reslut

@register.register
class Pad:
    def __init__(self):
        self.size_divisor = 32

    def __call__(self, result):
        return self.pad_image(result)

    def pad_image(self, result):
        image=result['img']
        height, width = image.shape[:2]
        target_width = int(np.ceil(width / self.size_divisor) * self.size_divisor)
        target_height = int(np.ceil(height / self.size_divisor) * self.size_divisor)
        pad_width = target_width - width
        pad_height = target_height - height
        image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        result['img']=image
        return result



