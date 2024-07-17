import torch
import torch.nn as nn
from collections import deque
# from quest.modules.v1 import *
import quest.utils.tensor_utils as TensorUtils
from quest.utils.utils import map_tensor_to_device
import quest.utils.obs_utils as ObsUtils
import einops


from abc import ABC, abstractmethod

class Policy(nn.Module, ABC):
    '''
    Super class with some basic functionality and functions we expect
    from all policy classes in our training loop
    '''

    def __init__(self, 
                 image_encoder_factory,
                 pointcloud_encoder_factory,
                 lowdim_encoder_factory,
                 image_aug_factory,
                 aug_factory,
                 obs_proj,
                 shape_meta,
                 device
                 ):
        super().__init__()
        
        self.use_image_augmentation = image_aug_factory is not None
        self.use_augmentation = aug_factory is not None
        if obs_proj is None:
            obs_proj = nn.Identity()
        self.obs_proj = obs_proj

        # observation encoders
        self.image_encoders, self.image_augs = {}, {}
        if image_encoder_factory is not None:
            for name, shape in shape_meta["observation"]['rgb'].items():
                self.image_encoders[name] = image_encoder_factory(shape)
                if self.use_image_augmentation:
                    self.image_augs[name] = image_aug_factory(input_shape=shape)
            self.image_encoders = nn.ModuleDict(self.image_encoders)
            self.image_augs = nn.ModuleDict(self.image_augs)
            # for name in shape_meta["observation"]['rgb'].items():
            #     self.image_encoders[name] = None
        
        self.lowdim_encoders = {}
        if lowdim_encoder_factory is not None:
            for name, shape in shape_meta['observation']['lowdim'].items():
                self.lowdim_encoders[name] = lowdim_encoder_factory(shape)
            self.lowdim_encoders = nn.ModuleDict(self.lowdim_encoders)
        
        self.pointcloud_encoders = {}
        if pointcloud_encoder_factory is not None:
            for name, shape in shape_meta['observation']['pointcloud'].items():
                self.pointcloud_encoders[name] = pointcloud_encoder_factory(shape)
            self.pointcloud_encoders = nn.ModuleDict(self.pointcloud_encoders)

        if self.use_augmentation:
            self.aug = aug_factory(shape_meta)

        # # add data augmentation for rgb inputs
        # self.image_aug = image_aug

        self.device = device

    @abstractmethod
    def compute_loss(self, data):
        raise NotImplementedError('Implement in subclass')

    @abstractmethod
    def get_optimizers(self):
        raise NotImplementedError('Implement in subclass')

    def get_schedulers(self, optimizers):
        return []
    
    def preprocess_input(self, data, train_mode=True):
        if train_mode and self.use_image_augmentation:  # apply augmentation
            # img_tuple = self._get_img_tuple(data)
            # aug_out = self._get_aug_output_dict(self.image_aug(img_tuple))
            # aug_out = {image_name: self.image_augs[image_name]()}
            for img_name in self.image_augs.keys():
                data["obs"][img_name] = self.image_augs[img_name](data['obs'][img_name])
        if train_mode and self.use_augmentation:
            data = self.aug(data)
        for key in self.image_encoders:
            x = TensorUtils.to_float(data['obs'][key])
            x = x / 255.
            x = torch.clip(x, 0, 1)
            data['obs'][key] = x
        return data

    def obs_encode(self, data, reduction='cat', hwc=False):
        ### 1. encode image
        # encoded = []
        img_encodings, pc_encodings, lowdim_encodings = [], [], []
        for img_name in self.image_encoders.keys():
            x = data["obs"][img_name]
            
            if hwc:
                x = einops.rearrange(x, 'B T H W C -> B T C H W')
            B, T, C, H, W = x.shape
            e = self.image_encoders[img_name](
                x.reshape(B * T, C, H, W),
                ).view(B, T, -1)
            img_encodings.append(e)
        for pointcloud_name in self.pointcloud_encoders.keys():
            x = data['obs'][pointcloud_name]
            e = self.pointcloud_encoders[pointcloud_name](x)
            pc_encodings.append(e)
        # 2. add proprio info
        for lowdim_name in self.lowdim_encoders.keys():
            lowdim_encodings.append(self.lowdim_encoders[lowdim_name](data["obs"][lowdim_name]))  # add (B, T, H_extra)

        if reduction == 'cat':
            encoded = img_encodings + pc_encodings + lowdim_encodings
            encoded = torch.cat(encoded, -1)  # (B, T, H_all)
            obs_emb = self.obs_proj(encoded) # TODO I feel that this projection should be algorithm-specific
        elif reduction == 'stack':
            encoded = img_encodings + pc_encodings + lowdim_encodings
            encoded = torch.stack(encoded, dim=2)
            obs_emb = self.obs_proj(encoded)
        elif reduction == 'none':
            return img_encodings, pc_encodings, lowdim_encodings
        return obs_emb
        # task_emb = self.task_encoder(data["task_id"]).unsqueeze(1)
        # if 
        # context = torch.cat([task_emb, init_obs_emb], dim=1)
        # return context

    def reset(self):
        return
    
    @abstractmethod
    def get_action(self, obs, task_id):
        raise NotImplementedError('Implement in subclass')
    

    # def _get_img_tuple(self, data):
    #     img_tuple = tuple(
    #         [data["obs"][img_name] for img_name in self.image_encoders.keys()]
    #     )
    #     return img_tuple

    # def _get_aug_output_dict(self, out):
    #     img_dict = {
    #         img_name: out[idx]
    #         for idx, img_name in enumerate(self.image_encoders.keys())
    #     }
    #     return img_dict
    
    def preprocess_dataset(self, dataset, use_tqdm=True):
        return


class ChunkPolicy(Policy):
    '''
    Super class for policies which predict chunks of actions
    '''
    def __init__(self, 
                 action_horizon,
                 **kwargs):
        super().__init__(**kwargs)

        self.action_horizon = action_horizon
        self.action_queue = None


    def reset(self):
        self.action_queue = deque(maxlen=self.action_horizon)
    
    def get_action(self, obs, task_id):
        assert self.action_queue is not None, "you need to call policy.reset() before getting actions"

        self.eval()
        if len(self.action_queue) == 0:
            for key, value in obs.items():
                if key in self.image_encoders:
                    value = ObsUtils.process_frame(value, channel_dim=3)
                obs[key] = torch.tensor(value).unsqueeze(0)
            batch = {}
            batch["obs"] = obs
            batch["task_id"] = torch.tensor([task_id], dtype=torch.long)
            batch = map_tensor_to_device(batch, self.device)

            with torch.no_grad():
                actions = self.sample_actions(batch).squeeze()
                self.action_queue.extend(actions[:self.action_horizon])
        action = self.action_queue.popleft()
        return action
    
    @abstractmethod
    def sample_actions(self, obs, task_id):
        raise NotImplementedError('Implement in subclass')

