import os
import glob
from typing import Optional, Union
import cv2
import numpy as np
import torch

from .Model import Model
from copy import deepcopy

from .compose import Compose

from deploy.data.scatter_gather import scatter
from .utils import read_json_file
import functools
from mmcv.parallel import collate, scatter
from torchvision import transforms
from deploy.transforms_3d import get_box_type


class fcos3d(Model):
    def __init__(self, config):
        super().__init__(config)
        self.agnostic = config['agnostic']
        self.strides = config['strides']
        self.image_width = config['IMAGE_WIDTH']
        self.image_height = config['IMAGE_HEIGHT']
        self.input_path = config['input_path']
        self.output_path = config['output_path']
        self.precision = config['precision']
        self.device = "cuda"
        print("device: %s" % self.device)
        self.test_pipeline = config['test_pipeline']
        self.dynamic_flag = False
        self.box_type_3d = config['box_type_3d']
        self.cam_intrinsic = config['cam_intrinsic']

    def compose(*functions):
        return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

    def create_input(self, imgs: Union[str, np.ndarray],
                     input_shape: None):
        test_pipeline = self.test_pipeline
        # static shape should ensure the shape before input image.
        if not self.dynamic_flag:
            transform = test_pipeline['pipeline'][1]
            if 'transforms' in transform:
                transform_list = transform['transforms']
                for i, step in enumerate(transform_list):
                    if step['type'] == 'Pad' and 'pad_to_square' in step \
                            and step['pad_to_square']:
                        transform_list.pop(i)
                        break
        # build the data pipeline
        test_pipeline = deepcopy(test_pipeline['pipeline'])
        test_pipeline = Compose(test_pipeline)
        # test_pipeline=transforms.Compose(test_pipeline)
        box_type_3d, box_mode_3d = get_box_type(self.box_type_3d)
        # get  info

        data = dict(
            image=imgs,
            img_info=dict(),
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d,
            img_fields=[],
            bbox3d_fields=[],
            pts_mask_fields=[],
            pts_seg_fields=[],
            bbox_fields=[],
            mask_fields=[],
            seg_fields=[])
        data['img_info'].update(
            dict(cam_intrinsic=self.cam_intrinsic))
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)

        data['img_metas'] = [
            img_metas.data[0] for img_metas in data['img_metas']
        ]
        data['img'] = [img.data[0] for img in data['img']]
        data['cam2img'] = [torch.tensor(data['img_metas'][0][0]['cam2img'])]
        data['cam2img_inverse'] = [torch.inverse(data['cam2img'][0])]
        if self.device != 'cpu':
            # scatter to specified GPU
            data = scatter(data, [self.device])[0]

        return data, tuple(data['img'] + data['cam2img'] +
                           data['cam2img_inverse'])

    def model_inference(self, arg):
        super().model_inference(arg)

    def detect(self, model, image):
        model_inputs, _ = self.create_input(image, image.shape)
        result = self.model_inference(model, model_inputs)[0]

    def post_proress(self):
        pass
        # TODO how to deal with result from detect

    def show_reslut(self):
        pass
        # todo how to show result in image
