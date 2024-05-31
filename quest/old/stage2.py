import torch
import torch.nn as nn
import numpy as np
from collections import deque
# from quest.modules.v1 import *
from utils.utils import torch_load_model
import robomimic.utils.tensor_utils as TensorUtils
from quest.modules.augmentation.data_augmentation import *
from quest.modules.rgb_modules.rgb_modules import ResnetEncoder

# def load_vae(cfg, tune_decoder):
#     skill_vae = SkillVAE(cfg)
#     if cfg.path is not None:
#         state_dict, _, _, _ = torch_load_model(cfg.path)
#         vae_state_dict = {key.replace('skill_vae.', ''): value for key, value in state_dict.items()}
#         skill_vae.load_state_dict(vae_state_dict, strict=True)
#     if not tune_decoder:
#         skill_vae.eval()
#         for param in skill_vae.parameters():
#             param.requires_grad = False
#     else:
#         skill_vae.train()
#     return skill_vae

class SkillGPT_Model(nn.Module):
    def __init__(self, cfg, shape_meta):
        super().__init__()
        policy_cfg = cfg.policy
        self.device = cfg.device
        self.use_augmentation = cfg.train.use_augmentation
        self.prior_cfg = policy_cfg.prior
        self.batch_size = cfg.train.batch_size
        self.start_token = policy_cfg.prior.start_token
        self.offset_loss_scale = policy_cfg.offset_loss_scale
        self.mpc_horizon = policy_cfg.mpc_horizon
        self.action_queue = deque(maxlen=self.mpc_horizon)
        self.act_dim = policy_cfg.action_dim
        self.vae_block_size = policy_cfg.skill_vae.skill_block_size
        self.return_offset = True if policy_cfg.prior.offset_layers > 0 else False
        offset_dim = self.act_dim*self.vae_block_size
        self.prior_cfg.offset_dim = offset_dim
        self.codebook_size = np.array(policy_cfg.skill_vae.fsq_level).prod()
        
        self.skill_vae = load_vae(policy_cfg.skill_vae, tune_decoder=cfg.tune_decoder).to(self.device)
        print(next(self.skill_vae.parameters()).requires_grad, 'skill_vae grad')
        self.skill_gpt = SkillGPT(self.prior_cfg).to(self.device)

        self.task_encodings = nn.Embedding(cfg.n_tasks, self.prior_cfg.n_embd)
        self.obs_proj = MLP_Proj(policy_cfg.cat_obs_dim, self.prior_cfg.n_embd, self.prior_cfg.n_embd)

        if cfg.train.loss_type == "mse":
            self.loss = torch.nn.MSELoss()
        elif cfg.train.loss_type == "l1":
            self.loss = torch.nn.L1Loss()
        else:
            raise NotImplementedError(f"Unknown loss type {cfg.train.loss_type}")
        
        # observation encoders
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
        init_obs_emb = self.obs_proj(encoded)
        task_emb = self.task_encodings(data["task_id"]).unsqueeze(1)
        context = torch.cat([task_emb, init_obs_emb], dim=1)
        return context


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

    # def preprocess_input(self, data, train_mode=True):
    #     if train_mode:  # apply augmentation
    #         if self.use_augmentation:
    #             img_tuple = self._get_img_tuple(data)
    #             aug_out = self._get_aug_output_dict(self.img_aug(img_tuple))
    #             for img_name in self.image_encoders.keys():
    #                 data["obs"][img_name] = aug_out[img_name]
    #         return data
    #     else:
    #         data = TensorUtils.recursive_dict_list_tuple_apply(
    #             data, {torch.Tensor: lambda x: x.unsqueeze(dim=1)}  # add time dimension
    #         )
    #         data["task_id"] = data["task_id"].squeeze(1)
    #     return data

    # def forward(self, data):
    #     with torch.no_grad():
    #         indices = self.skill_vae.get_indices(data["actions"]).long()
    #     context = self.obs_encode(data)
    #     start_tokens = (torch.ones((context.shape[0], 1))*self.start_token).long().to(self.device)
    #     x = torch.cat([start_tokens, indices[:,:-1]], dim=1)
    #     targets = indices.clone()
    #     logits, prior_loss, offset = self.skill_gpt(x, context, targets, return_offset=self.return_offset)
    #     with torch.no_grad():
    #         logits = logits[:,:,:self.codebook_size]
    #         probs = torch.softmax(logits, dim=-1)
    #         sampled_indices = torch.multinomial(probs.view(-1,logits.shape[-1]),1)
    #         sampled_indices = sampled_indices.view(-1,logits.shape[1])
    #     pred_actions = self.skill_vae.decode_actions(sampled_indices)
    #     if self.return_offset:
    #         offset = offset.view(-1, self.vae_block_size, self.act_dim)
    #         pred_actions = pred_actions + offset
    #     offset_loss = self.loss(pred_actions, data["actions"])
    #     total_loss = prior_loss + self.offset_loss_scale*offset_loss
    #     return total_loss, {'offset_loss': offset_loss}

    # def compute_loss(self, data):
    #     data = self.preprocess_input(data, train_mode=True)
    #     loss, info = self.forward(data)
    #     return loss, info
    
    # def get_action(self, data):
    #     self.eval()
    #     if len(self.action_queue) == 0:
    #         with torch.no_grad():
    #             actions = self.sample_actions(data)
    #             self.action_queue.extend(actions[:self.mpc_horizon])
    #     action = self.action_queue.popleft()
    #     return action
    
    # def get_indices_top_k(self, context):
    #     x = torch.ones((context.shape[0], 1)).long().to(self.device)*self.start_token
    #     for i in range(self.prior_cfg.block_size):
    #         if i == self.prior_cfg.block_size-1:
    #             logits,offset = self.skill_gpt(x, context, return_offset=self.return_offset)
    #             logits = logits[:,:,:self.codebook_size]
    #             offset = offset.view(-1, self.vae_block_size, self.act_dim) if self.return_offset else None
    #         else:
    #             logits,_ = self.skill_gpt(x, context)
    #             logits = logits[:,:,:self.codebook_size]
    #         next_indices = top_k_sampling(logits[:,-1,:], self.prior_cfg.beam_size, self.prior_cfg.temperature)
    #         x = torch.cat([x, next_indices], dim=1)
    #     return x[:,1:], offset
    #
    # def sample_actions(self, data):
    #     data = self.preprocess_input(data, train_mode=False)
    #     context = self.obs_encode(data)
    #     sampled_indices, offset = self.get_indices_top_k(context)
    #     pred_actions = self.skill_vae.decode_actions(sampled_indices)
    #     pred_actions_with_offset = pred_actions + offset if offset is not None else pred_actions
    #     pred_actions_with_offset = pred_actions_with_offset.permute(1,0,2)
    #     return pred_actions_with_offset.detach().cpu().numpy()