import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from quest.algos.utils.obs_core import CropRandomizer
from quest.utils.mujoco_point_cloud import batch_axis_angle_to_rotation_matrix

# from pyinstrument import Profiler
# profiler = Profiler()


class IdentityAug(nn.Module):
    def __init__(self, input_shape=None, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

    def output_shape(self, input_shape):
        return input_shape


class TranslationAug(nn.Module):
    """
    Utilize the random crop from robomimic.
    """

    def __init__(
        self,
        input_shape,
        translation,
    ):
        super().__init__()

        self.pad_translation = translation // 2
        pad_output_shape = (
            input_shape[0],
            input_shape[1] + translation,
            input_shape[2] + translation,
        )

        self.crop_randomizer = CropRandomizer(
            input_shape=pad_output_shape,
            crop_height=input_shape[1],
            crop_width=input_shape[2],
        )

    def forward(self, x):
        if self.training:
            batch_size, temporal_len, img_c, img_h, img_w = x.shape
            x = x.reshape(batch_size, temporal_len * img_c, img_h, img_w)
            out = F.pad(x, pad=(self.pad_translation,) * 4, mode="replicate")
            out = self.crop_randomizer.forward_in(out)
            out = out.reshape(batch_size, temporal_len, img_c, img_h, img_w)
        else:
            out = x
        return out

    def output_shape(self, input_shape):
        return input_shape


class ImgColorJitterAug(torch.nn.Module):
    """
    Conduct color jittering augmentation outside of proposal boxes
    """

    def __init__(
        self,
        input_shape,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        epsilon=0.05,
    ):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.epsilon = epsilon

    def forward(self, x):
        if self.training and np.random.rand() > self.epsilon:
            out = self.color_jitter(x)
        else:
            out = x
        return out

    def output_shape(self, input_shape):
        return input_shape


class ImgColorJitterGroupAug(torch.nn.Module):
    """
    Conduct color jittering augmentation outside of proposal boxes
    """

    def __init__(
        self,
        input_shape,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        epsilon=0.05,
    ):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.epsilon = epsilon

    def forward(self, x):
        raise NotImplementedError
        if self.training and np.random.rand() > self.epsilon:
            out = self.color_jitter(x)
        else:
            out = x
        return out

    def output_shape(self, input_shape):
        return input_shape


class BatchWiseImgColorJitterAug(torch.nn.Module):
    """
    Color jittering augmentation to individual batch.
    This is to create variation in training data to combat
    BatchNorm in convolution network.
    """

    def __init__(
        self,
        input_shape,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        epsilon=0.1,
    ):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.epsilon = epsilon

    def forward(self, x):
        mask = torch.rand((x.shape[0], *(1,)*(len(x.shape)-1)), device=x.device) > self.epsilon
        
        # profiler.start()
        jittered = self.color_jitter(x)
        # profiler.stop()
        # profiler.print(show_all=True)

        out = mask * jittered + torch.logical_not(mask) * x
        return out

    def output_shape(self, input_shape):
        return input_shape


class DataAugGroup(nn.Module):
    """
    Add augmentation to multiple inputs
    """

    def __init__(self, aug_list, input_shape):
        super().__init__()
        aug_list = [aug(input_shape) for aug in aug_list]
        self.aug_layer = nn.Sequential(*aug_list)

    def forward(self, x):
        return self.aug_layer(x)
    

class PointcloudRotationAug(nn.Module):
    def __init__(self, shape_meta, output_frame='hand', action_space='world'):
        super().__init__()

        assert output_frame in ('hand', 'world')
        self.output_frame = output_frame

        assert action_space in ('hand', 'world')
        self.action_space = action_space

    def forward(self, data):
        obs_data = data['obs']
        pointcloud = obs_data['point_cloud']
        hand_mat = obs_data['hand_mat']
        hand_mat_inv = obs_data['hand_mat_inv']
        actions = data['actions']
        B = actions.shape[0]

        theta = torch.rand(B, device=actions.device)
        axes = torch.tensor([(1, 0, 0)] * B, device=actions.device)
        rot_mats = batch_axis_angle_to_rotation_matrix(axes, theta)

        # hand_mat_inv = torch.linalg.inv()
