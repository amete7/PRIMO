import torch
import torch.nn as nn
import numpy as np
import einops
import primo
from collections import deque
from utils.utils import torch_load_model
import robomimic.utils.tensor_utils as TensorUtils
from primo.modules.augmentation.data_augmentation import *
from primo.modules.rgb_modules.rgb_modules import ResnetEncoder
from primo.modules.v1 import MLP_Proj
from primo.modules.diffusion_modules import Diffusion_Model

class Diffusion_Policy(nn.Module):
    def __init__(self, cfg, shape_meta):
        super().__init__(cfg)
        policy_cfg = cfg.policy
        self.device = cfg.device
        self.use_augmentation = cfg.train.use_augmentation
        self.mpc_horizon = policy_cfg.mpc_horizon
        self.action_queue = deque(maxlen=self.mpc_horizon)
        
        self.diff_model = Diffusion_Model(policy_cfg, self.device)
        self.diff_model = self.diff_model.to(self.device)
        # self.input_proj = MLP_Proj(policy_cfg.cat_obs_dim+policy_cfg.lang_emb_dim, policy_cfg.cond_dim, policy_cfg.cond_dim)

        self.image_encoders = {}

        for name in shape_meta["all_shapes"].keys():
            if "rgb" in name or "depth" in name:
                kwargs = policy_cfg.image_encoder.network_kwargs
                kwargs.input_shape = shape_meta["all_shapes"][name]
                kwargs.output_size = policy_cfg.obs_emb_dim
                self.image_encoders[name] = {
                    "input_shape": shape_meta["all_shapes"][name],
                    "encoder": eval(policy_cfg.image_encoder.network)(**kwargs),
                }
        self.encoders = nn.ModuleList(
            [x["encoder"] for x in self.image_encoders.values()]
        )

        self.proprio_encoder = MLP_Proj(shape_meta["all_shapes"]['robot_states'][0], policy_cfg.proprio_emb_dim, policy_cfg.proprio_emb_dim)
        self.task_encodings = nn.Embedding(cfg.n_tasks, self.prior_cfg.n_embd)

            # add data augmentation for rgb inputs
        color_aug = eval(policy_cfg.color_aug.network)(
            **policy_cfg.color_aug.network_kwargs
        )
        policy_cfg.translation_aug.network_kwargs["input_shape"] = shape_meta[
            "all_shapes"
        ][cfg.data.obs.modality.rgb[0]]
        translation_aug = eval(policy_cfg.translation_aug.network)(
            **policy_cfg.translation_aug.network_kwargs
        )
        self.img_aug = DataAugGroup((color_aug, translation_aug))
        # self.encoders.append(self.extra_encoder)

    def obs_encode(self, data):
        ### 1. encode image
        encoded = []
        for img_name in self.image_encoders.keys():
            x = data["obs"][img_name]
            B, T, C, H, W = x.shape
            e = self.image_encoders[img_name]["encoder"](
                x.reshape(B * T, C, H, W),
                langs=None,
            ).view(B, T, -1)
            encoded.append(e)
        # 2. add gripper info
        encoded.append(self.proprio_encoder(data["obs"]['robot_states']))  # add (B, T, H_extra)
        encoded = torch.cat(encoded, -1)  # (B, T, H_all)
        return encoded.squeeze(1)

    def forward(self, data):
        init_obs = self.obs_encode(data)
        lang_emb = self.task_encodings(data["task_id"]).unsqueeze(0)
        cond = torch.cat([init_obs, lang_emb], dim=-1)
        loss = self.diff_model(cond,data["actions"])
        return loss

    def compute_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        loss = self.forward(data)
        return loss, {}
    
    def get_action(self, data):
        self.eval()
        if len(self.action_queue) == 0:
            with torch.no_grad():
                actions = self.sample_actions(data)
                self.action_queue.extend(actions[:self.mpc_horizon])
        action = self.action_queue.popleft()
        return action
    
    def sample_actions(self, data):
        data = self.preprocess_input(data, train_mode=False)
        init_obs = self.obs_encode(data)
        lang_emb = data["task_emb"]
        cond = torch.cat([init_obs, lang_emb], dim=-1)
        actions = self.diff_model.get_action(cond)
        actions = actions.permute(1,0,2)
        return actions.detach().cpu().numpy()

    def reset(self):
        self.action_queue = deque(maxlen=self.mpc_horizon)
        
    def _get_img_tuple(self, data):
        img_tuple = tuple(
            [data["obs"][img_name] for img_name in self.image_encoders.keys()]
        )
        return img_tuple

    def _get_aug_output_dict(self, out):
        img_dict = {
            img_name: out[idx]
            for idx, img_name in enumerate(self.image_encoders.keys())
        }
        return img_dict

    def preprocess_input(self, data, train_mode=True):
        if train_mode:  # apply augmentation
            if self.use_augmentation:
                img_tuple = self._get_img_tuple(data)
                aug_out = self._get_aug_output_dict(self.img_aug(img_tuple))
                for img_name in self.image_encoders.keys():
                    data["obs"][img_name] = aug_out[img_name]
            return data
        else:
            data = TensorUtils.recursive_dict_list_tuple_apply(
                data, {torch.Tensor: lambda x: x.unsqueeze(dim=1)}  # add time dimension
            )
            data["task_id"] = data["task_id"].squeeze(1)
        return data