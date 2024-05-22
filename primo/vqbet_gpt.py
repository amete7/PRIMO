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
from primo.vqbet_vae import VQVAE_Model
from primo.vqbet_modules.vq_behavior_transformer.bet import BehaviorTransformer
from primo.vqbet_modules.vq_behavior_transformer.gpt import GPT, GPTConfig


class VQBet_Model(nn.Module):
    def __init__(self, cfg, shape_meta):
        super().__init__()
        policy_cfg = cfg.policy
        self.device = cfg.device
        self.use_augmentation = cfg.train.use_augmentation
        self.mpc_horizon = policy_cfg.mpc_horizon
        self.obs_window_size = policy_cfg.obs_window_size
        self.action_queue = deque(maxlen=self.mpc_horizon)
        
        vq_vae = VQVAE_Model(cfg)
        if policy_cfg.vqvae_path is not None:
            vq_vae.load_state_dict(torch_load_model(policy_cfg.vqvae_path)[0])
        vq_vae = vq_vae.to(self.device)
        if not cfg.tune_decoder:
            vq_vae.eval()
            for param in vq_vae.parameters():
                param.requires_grad = False
        else:
            vq_vae.train()
        
        gpt = GPT(GPTConfig(
            block_size=policy_cfg.gpt_block_size,
            input_dim=policy_cfg.gpt_n_embd,
            n_layer=policy_cfg.gpt_n_layer,
            n_head=policy_cfg.gpt_n_head,
            n_embd=policy_cfg.gpt_n_embd,
        )).to(self.device)

        self.Bet = BehaviorTransformer(
            gpt_model=gpt,
            vqvae_model=vq_vae.vq_vae,
            obs_dim=policy_cfg.gpt_n_embd,
            act_dim=policy_cfg.action_dim,
            goal_dim=policy_cfg.gpt_n_embd,
            obs_window_size=policy_cfg.obs_window_size,
            act_window_size=policy_cfg.skill_block_size,
            offset_loss_multiplier=policy_cfg.offset_loss_multiplier,
        ).to(self.device)

        self.task_encodings = nn.Embedding(cfg.n_tasks, policy_cfg.gpt_n_embd)
        self.obs_proj = MLP_Proj(policy_cfg.cat_obs_dim, policy_cfg.gpt_n_embd, policy_cfg.gpt_n_embd)
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
        self.proprio_encoder = MLP_Proj(shape_meta["all_shapes"]['robot_states'][0], policy_cfg.proprio_emb_dim, policy_cfg.proprio_emb_dim)
        self.encoders = nn.ModuleList(
            [x["encoder"] for x in self.image_encoders.values()]
        )
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
                ).view(B, T, -1)
            encoded.append(e)
        # 2. add proprio info
        encoded.append(self.proprio_encoder(data["obs"]['robot_states']))  # add (B, T, H_extra)
        encoded = torch.cat(encoded, -1)  # (B, T, H_all)
        return encoded

    def forward(self, data):
        obs = self.obs_proj(self.obs_encode(data))
        obs = obs[:,:self.obs_window_size,:]
        goal = self.task_encodings(data["task_id"]).unsqueeze(1)
        predicted_act, loss, loss_dict = self.Bet(obs, goal, data["actions"])
        return predicted_act, loss, loss_dict

    def compute_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        _, loss, loss_dict = self.forward(data)
        return loss, loss_dict
    
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
        obs = self.obs_proj(self.obs_encode(data))[:,:self.obs_window_size,:]
        goal = self.task_encodings(data["task_id"]).unsqueeze(1)
        predicted_act, _, _ = self.Bet(obs, goal, None)
        predicted_act = einops.rearrange(predicted_act, "(N T) W A -> N T W A", T=self.obs_window_size)[:, -1, :, :]
        predicted_act = predicted_act.permute(1,0,2)
        return predicted_act.detach().cpu().numpy()

    def reset(self):
        self.action_queue = deque(maxlen=self.mpc_horizon)
    
    def configure_optimizers(self, lr, betas, weight_decay):
        bet_optimizers = self.Bet.configure_optimizers(weight_decay=weight_decay, learning_rate=lr, betas=betas)
        bet_optimizers['optimizer1'].add_param_group({'params': self.task_encodings.parameters()})
        bet_optimizers['optimizer1'].add_param_group({'params': self.obs_proj.parameters()})
        bet_optimizers['optimizer1'].add_param_group({'params': self.encoders.parameters(), 'lr': lr*0.1})
        bet_optimizers['optimizer1'].add_param_group({'params': self.proprio_encoder.parameters()})
        return bet_optimizers

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