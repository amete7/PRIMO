import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import deque

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
from quest.algos.baseline_modules.vq_behavior_transformer.gpt import GPT
from quest.algos.baseline_modules.vq_behavior_transformer.utils import MLP
from quest.algos.baseline_modules.vq_behavior_transformer.vqvae import VqVae
from quest.algos.utils.mlp_proj import MLPProj
import quest.utils.tensor_utils as TensorUtils

from quest.utils.utils import map_tensor_to_device
import quest.utils.obs_utils as ObsUtils

class BehaviorTransformer(nn.Module):
    GOAL_SPEC = Enum("GOAL_SPEC", "concat stack unconditional")

    def __init__(
        self,
        autoencoder,
        policy_prior,
        stage,
        optimizer_config,
        optimizer_factory,
        image_encoder_factory,
        proprio_encoder,
        image_aug,
        loss_fn,
        n_tasks,
        cat_obs_dim,
        # l1_loss_scale,
        action_horizon,
        shape_meta, 
        offset_loss_multiplier: float = 1.0e3,
        secondary_code_multiplier: float = 0.5,

        obs_window_size=10,
        skill_block_size=10,
        sequentially_select=False,
        device=None,
        # visual_input=False,
        # finetune_resnet=False,
    ):
        super().__init__()
        # self._obs_dim = obs_dim

        self.autoencoder = autoencoder
        self.policy_prior = policy_prior
        self.stage = stage
        self.use_augmentation = image_aug is not None
        self.optimizer_config = optimizer_config
        self.optimizer_factory = optimizer_factory

        self.obs_window_size = obs_window_size
        self.skill_block_size = skill_block_size
        self.sequentially_select = sequentially_select
        self.action_horizon = action_horizon
        # if goal_dim <= 0:
        #     self._cbet_method = self.GOAL_SPEC.unconditional
        # elif obs_dim == goal_dim:
        
        # TODO maybe unhardcode this (maybe)
        self._cbet_method = self.GOAL_SPEC.concat
        # else:
        #     self._cbet_method = self.GOAL_SPEC.stack
        print("initialize VQ-BeT agent")


        self.task_encodings = nn.Embedding(n_tasks, self.policy_prior.n_embd)
        self.obs_proj = MLPProj(cat_obs_dim, self.policy_prior.n_embd)
        
        # add data augmentation for rgb inputs
        self.image_aug = image_aug

        # observation encoders
        image_encoders = {}
        for name in shape_meta["image_inputs"]:
            image_encoders[name] = image_encoder_factory()
        self.image_encoders = nn.ModuleDict(image_encoders)
        self.proprio_encoder = proprio_encoder

        # if visual_input:
        #     self._resnet_header = MLP(
        #         in_channels=512,
        #         hidden_channels=[1024],
        #     )
        self._offset_loss_multiplier = offset_loss_multiplier
        self._secondary_code_multiplier = secondary_code_multiplier
        self._criterion = loss_fn
        # self.visual_input = visual_input
        # self.finetune_resnet = finetune_resnet

        # For now, we assume the number of clusters is given.

        self._G = self.autoencoder.vqvae_groups  # G(number of groups)
        self._C = self.autoencoder.vqvae_n_embed  # C(number of code integers)
        self._D = self.autoencoder.embedding_dim  # D(embedding dims)

        if self.sequentially_select:
            print("use sequantial prediction for vq dictionary!")
            self._map_to_cbet_preds_bin1 = MLP(
                in_channels=policy_prior.output_dim,
                hidden_channels=[512, 512, self._C],
            )
            self._map_to_cbet_preds_bin2 = MLP(
                in_channels=policy_prior.output_dim + self._C,
                hidden_channels=[512, self._C],
            )
        else:
            self._map_to_cbet_preds_bin = MLP(
                in_channels=policy_prior.output_dim,
                hidden_channels=[1024, 1024, self._G * self._C],
            )
        self._map_to_cbet_preds_offset = MLP(
            in_channels=policy_prior.output_dim,
            hidden_channels=[
                1024,
                1024,
                self._G * self._C * (shape_meta.action_dim * self.skill_block_size),
            ],
        )

        self.action_queue = None
        self.device = device

        # if visual_input:
        #     import torchvision.models as models
        #     import torchvision.transforms as transforms

        #     resnet = models.resnet18(pretrained=True)
        #     self.resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])).cuda()
        #     if not self.finetune_resnet:
        #         for param in self.resnet.parameters():
        #             param.requires_grad = False
        #     self.transform = transforms.Compose(
        #         [  # transforms.Resize((224, 224)), \
        #             # if error -> delete Resize and add F.interpolate before applying self.transform.
        #             # TODO np to tensor
        #             transforms.Normalize(
        #                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255]
        #             )
        #         ]
        #     )

    # def forward(self, data) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    #     obs = self.obs_proj(self.obs_encode(data))
    #     obs = obs[:,:self.obs_window_size,:]
    #     goal = self.task_encodings(data["task_id"]).unsqueeze(1)
    #     return self._predict(obs, goal, data['actions'])

    def compute_loss(self, data):
        if self.stage == 0:
            return self.compute_autoencoder_loss(data)
        elif self.stage == 1:
            return self.compute_prior_loss(data)
        elif self.stage == 2:
            pass

    def compute_autoencoder_loss(self, data):
        pred, total_loss, l1_loss, codebook_loss, pp = self.autoencoder(data["actions"])
        info = {
            'autoencoder/recon_loss': l1_loss.item(), 
            'autoencoder/codebook_loss': codebook_loss.item(), 
            'autoencoder/pp': pp}
        return total_loss, info
    
    def compute_prior_loss(self, data):
        data = self.preprocess_input(data)

        context = self.obs_encode(data)
        predicted_action, decoded_action, sampled_centers, logit_info = self._predict(context)

        action_seq = data['actions']
        n, total_w, act_dim = action_seq.shape
        act_w = self.autoencoder.input_dim_h
        obs_w = total_w + 1 - act_w
        output_shape = (n, obs_w, act_w, act_dim)
        output = torch.empty(output_shape, device=action_seq.device)
        for i in range(obs_w):
            output[:, i, :, :] = action_seq[:, i : i + act_w, :]
        action_seq = einops.rearrange(output, "N T W A -> (N T) W A")
        NT = action_seq.shape[0]
        # Figure out the loss for the actions.
        # First, we need to find the closest cluster center for each action.
        state_vq, action_bins = self.autoencoder.get_code(
            action_seq
        )  # action_bins: NT, G

        # Now we can compute the loss.
        if action_seq.ndim == 2:
            action_seq = action_seq.unsqueeze(0)

        offset_loss = torch.nn.L1Loss()(action_seq, predicted_action)

        action_diff = F.mse_loss(
            einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
                :, -1, 0, :
            ],
            einops.rearrange(predicted_action, "(N T) W A -> N T W A", T=obs_w)[
                :, -1, 0, :
            ],
        )  # batch, time, windowsize (t ... t+N), action dim -> [:, -1, 0, :] is for rollout
        action_diff_tot = F.mse_loss(
            einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
                :, -1, :, :
            ],
            einops.rearrange(predicted_action, "(N T) W A -> N T W A", T=obs_w)[
                :, -1, :, :
            ],
        )  # batch, time, windowsize (t ... t+N), action dim -> [:, -1, 0, :] is for rollout
        action_diff_mean_res1 = (
            abs(
                einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
                    :, -1, 0, :
                ]
                - einops.rearrange(decoded_action, "(N T) W A -> N T W A", T=obs_w)[
                    :, -1, 0, :
                ]
            )
        ).mean()
        action_diff_mean_res2 = (
            abs(
                einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
                    :, -1, 0, :
                ]
                - einops.rearrange(
                    predicted_action, "(N T) W A -> N T W A", T=obs_w
                )[:, -1, 0, :]
            )
        ).mean()
        action_diff_max = (
            abs(
                einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
                    :, -1, 0, :
                ]
                - einops.rearrange(
                    predicted_action, "(N T) W A -> N T W A", T=obs_w
                )[:, -1, 0, :]
            )
        ).max()

        if self.sequentially_select:
            cbet_logits1, gpt_output = logit_info
            cbet_loss1 = self._criterion(  # F.cross_entropy
                cbet_logits1[:, :],
                action_bins[:, 0],
            )
            cbet_logits2 = self._map_to_cbet_preds_bin2(
                torch.cat(
                    (gpt_output, F.one_hot(action_bins[:, 0], num_classes=self._C)),
                    axis=1,
                )
            )
            cbet_loss2 = self._criterion(  # F.cross_entropy
                cbet_logits2[:, :],
                action_bins[:, 1],
            )
        else:
            cbet_logits = logit_info
            cbet_loss1 = self._criterion(  # F.cross_entropy
                cbet_logits[:, 0, :],
                action_bins[:, 0],
            )
            cbet_loss2 = self._criterion(  # F.cross_entropy
                cbet_logits[:, 1, :],
                action_bins[:, 1],
            )
        cbet_loss = cbet_loss1 * 5 + cbet_loss2 * self._secondary_code_multiplier

        equal_total_code_rate = (
            torch.sum(
                (
                    torch.sum((action_bins == sampled_centers).int(), axis=1) == self._G
                ).int()
            )
            / NT
        )
        equal_single_code_rate = torch.sum(
            (action_bins[:, 0] == sampled_centers[:, 0]).int()
        ) / (NT)
        equal_single_code_rate2 = torch.sum(
            (action_bins[:, 1] == sampled_centers[:, 1]).int()
        ) / (NT)

        loss = cbet_loss + self._offset_loss_multiplier * offset_loss
        info = {
            "prior/classification_loss": cbet_loss.detach().cpu().item(),
            "prior/offset_loss": offset_loss.detach().cpu().item(),
            "prior/total_loss": loss.detach().cpu().item(),
            "prior/equal_total_code_rate": equal_total_code_rate.item(),
            "prior/equal_single_code_rate": equal_single_code_rate.item(),
            "prior/equal_single_code_rate2": equal_single_code_rate2.item(),
            "prior/action_diff": action_diff.detach().cpu().item(),
            "prior/action_diff_tot": action_diff_tot.detach().cpu().item(),
            "prior/action_diff_mean_res1": action_diff_mean_res1.detach().cpu().item(),
            "prior/action_diff_mean_res2": action_diff_mean_res2.detach().cpu().item(),
            "prior/action_diff_max": action_diff_max.detach().cpu().item(),
        }
        return loss, info

    def _predict(
        self,
        gpt_input):
        # Assume dimensions are N T D for N sequences of T timesteps with dimension D.
        # if self.visual_input:
        #     obs_seq = obs_seq.cuda()
        #     if obs_seq.ndim == 3:
        #         obs_seq = obs_seq.clone().detach()
        #         obs_seq = self._resnet_header(obs_seq)
        #     else:
        #         N = obs_seq.shape[0]
        #         if obs_seq.shape[-1] == 3:
        #             obs_seq = (
        #                 einops.rearrange(obs_seq, "N T W H C -> (N T) C W H") / 255.0
        #             )  # * 2. - 1.
        #         else:
        #             obs_seq = (
        #                 einops.rearrange(obs_seq, "N T C W H -> (N T) C W H") / 255.0
        #             )  # * 2. - 1.
        #         # obs_seq = obs_seq.cuda()
        #         if obs_seq.shape[-1] != 224:
        #             obs_seq = F.interpolate(obs_seq, size=224)
        #         obs_seq = self.transform(obs_seq)
        #         obs_seq = torch.squeeze(torch.squeeze(self.resnet(obs_seq), -1), -1)
        #         obs_seq = self._resnet_header(obs_seq)
        #         obs_seq = einops.rearrange(obs_seq, "(N T) L -> N T L", N=N)
        #     if not (self._cbet_method == self.GOAL_SPEC.unconditional):
        #         goal_seq = goal_seq.cuda()
        #         if goal_seq.ndim == 3:
        #             goal_seq = goal_seq.clone().detach()
        #             goal_seq = self._resnet_header(goal_seq)
        #         else:
        #             if goal_seq.shape[-1] == 3:
        #                 goal_seq = (
        #                     einops.rearrange(goal_seq, "N T W H C -> (N T) C W H")
        #                     / 255.0
        #                 )  # * 2. - 1.
        #             else:
        #                 goal_seq = (
        #                     einops.rearrange(goal_seq, "N T C W H -> (N T) C W H")
        #                     / 255.0
        #                 )  # * 2. - 1.
        #             # goal_seq = goal_seq.cuda()
        #             if goal_seq.shape[-1] != 224:
        #                 goal_seq = F.interpolate(goal_seq, size=224)
        #             goal_seq = self.transform(goal_seq)
        #             goal_seq = torch.squeeze(
        #                 torch.squeeze(self.resnet(goal_seq), -1), -1
        #             )
        #             goal_seq = self._resnet_header(goal_seq)
        #             goal_seq = einops.rearrange(goal_seq, "(N T) L -> N T L", N=N)
        # if obs_seq.shape[1] < self.obs_window_size:
        #     obs_seq = torch.cat(
        #         (
        #             torch.tile(
        #                 obs_seq[:, 0, :].unsqueeze(1),
        #                 (1, self.obs_window_size - obs_seq.shape[1], 1),
        #             ),
        #             obs_seq,
        #         ),
        #         dim=-2,
        #     )
        # if self._cbet_method == self.GOAL_SPEC.unconditional:
        #     gpt_input = obs_seq
        # elif self._cbet_method == self.GOAL_SPEC.concat:
        #     gpt_input = torch.cat([goal_seq, obs_seq], dim=1)
        # elif self._cbet_method == self.GOAL_SPEC.stack:
        #     gpt_input = torch.cat([goal_seq, obs_seq], dim=-1)
        # else:
        #     raise NotImplementedError
        gpt_output = self.policy_prior(gpt_input)
        # breakpoint()

        # if self._cbet_method == self.GOAL_SPEC.unconditional:
        #     gpt_output = gpt_output
        # else:
        #     gpt_output = gpt_output[:, goal_seq.size(1) :, :]

        # TODO: this might cause some bugs
        gpt_output = gpt_output[:, 1:, :]

        gpt_output = einops.rearrange(gpt_output, "N T (G C) -> (N T) (G C)", G=self._G)
        # obs = einops.rearrange(obs_seq, "N T O -> (N T) O")
        # obs = obs.unsqueeze(dim=1)

        if self.sequentially_select:
            cbet_logits1 = self._map_to_cbet_preds_bin1(gpt_output)
            cbet_offsets = self._map_to_cbet_preds_offset(gpt_output)
            cbet_offsets = einops.rearrange(
                cbet_offsets, "(NT) (G C WA) -> (NT) G C WA", G=self._G, C=self._C
            )
            cbet_probs1 = torch.softmax(cbet_logits1, dim=-1)
            NT, choices = cbet_probs1.shape
            G = self._G
            sampled_centers1 = einops.rearrange(
                torch.multinomial(cbet_probs1.view(-1, choices), num_samples=1),
                "(NT) 1 -> NT",
                NT=NT,
            )
            cbet_logits2 = self._map_to_cbet_preds_bin2(
                torch.cat(
                    (gpt_output, F.one_hot(sampled_centers1, num_classes=self._C)),
                    axis=1,
                )
            )
            cbet_probs2 = torch.softmax(cbet_logits2, dim=-1)
            sampled_centers2 = einops.rearrange(
                torch.multinomial(cbet_probs2.view(-1, choices), num_samples=1),
                "(NT) 1 -> NT",
                NT=NT,
            )
            sampled_centers = torch.stack(
                (sampled_centers1, sampled_centers2), axis=1
            )  # NT, G
        else:
            cbet_logits = self._map_to_cbet_preds_bin(gpt_output)
            cbet_offsets = self._map_to_cbet_preds_offset(gpt_output)
            cbet_logits = einops.rearrange(
                cbet_logits, "(NT) (G C) -> (NT) G C", G=self._G
            )
            cbet_offsets = einops.rearrange(
                cbet_offsets, "(NT) (G C WA) -> (NT) G C WA", G=self._G, C=self._C
            )
            cbet_probs = torch.softmax(cbet_logits, dim=-1)
            NT, G, choices = cbet_probs.shape
            sampled_centers = einops.rearrange(
                torch.multinomial(cbet_probs.view(-1, choices), num_samples=1),
                "(NT G) 1 -> NT G",
                NT=NT,
            )

        indices = (
            torch.arange(NT).unsqueeze(1).cuda(),
            torch.arange(self._G).unsqueeze(0).cuda(),
            sampled_centers,
        )
        # Use advanced indexing to sample the values
        sampled_offsets = cbet_offsets[indices]  # NT, G, W, A(?) or NT, G, A

        sampled_offsets = sampled_offsets.sum(dim=1)
        centers = self.autoencoder.draw_code_forward(sampled_centers).view(
            NT, -1, self._D
        )
        return_decoder_input = einops.rearrange(
            centers.clone().detach(), "NT G D -> NT (G D)"
        )
        decoded_action = (
            self.autoencoder.get_action_from_latent(return_decoder_input)
            .clone()
            .detach()
        )  # NT, A
        sampled_offsets = einops.rearrange(
            sampled_offsets, "NT (W A) -> NT W A", W=self.autoencoder.input_dim_h
        )
        predicted_action = decoded_action + sampled_offsets

        if self.sequentially_select:
            return predicted_action, decoded_action, sampled_centers, (cbet_logits1, gpt_output)
        return predicted_action, decoded_action, sampled_centers, cbet_logits
        # if action_seq is not None:
        #     n, total_w, act_dim = action_seq.shape
        #     act_w = self.autoencoder.input_dim_h
        #     obs_w = total_w + 1 - act_w
        #     output_shape = (n, obs_w, act_w, act_dim)
        #     output = torch.empty(output_shape).to(action_seq.device)
        #     for i in range(obs_w):
        #         output[:, i, :, :] = action_seq[:, i : i + act_w, :]
        #     action_seq = einops.rearrange(output, "N T W A -> (N T) W A")
        #     # Figure out the loss for the actions.
        #     # First, we need to find the closest cluster center for each action.
        #     state_vq, action_bins = self.autoencoder.get_code(
        #         action_seq
        #     )  # action_bins: NT, G

        #     # Now we can compute the loss.
        #     if action_seq.ndim == 2:
        #         action_seq = action_seq.unsqueeze(0)

        #     offset_loss = torch.nn.L1Loss()(action_seq, predicted_action)

        #     action_diff = F.mse_loss(
        #         einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
        #             :, -1, 0, :
        #         ],
        #         einops.rearrange(predicted_action, "(N T) W A -> N T W A", T=obs_w)[
        #             :, -1, 0, :
        #         ],
        #     )  # batch, time, windowsize (t ... t+N), action dim -> [:, -1, 0, :] is for rollout
        #     action_diff_tot = F.mse_loss(
        #         einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
        #             :, -1, :, :
        #         ],
        #         einops.rearrange(predicted_action, "(N T) W A -> N T W A", T=obs_w)[
        #             :, -1, :, :
        #         ],
        #     )  # batch, time, windowsize (t ... t+N), action dim -> [:, -1, 0, :] is for rollout
        #     action_diff_mean_res1 = (
        #         abs(
        #             einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
        #                 :, -1, 0, :
        #             ]
        #             - einops.rearrange(decoded_action, "(N T) W A -> N T W A", T=obs_w)[
        #                 :, -1, 0, :
        #             ]
        #         )
        #     ).mean()
        #     action_diff_mean_res2 = (
        #         abs(
        #             einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
        #                 :, -1, 0, :
        #             ]
        #             - einops.rearrange(
        #                 predicted_action, "(N T) W A -> N T W A", T=obs_w
        #             )[:, -1, 0, :]
        #         )
        #     ).mean()
        #     action_diff_max = (
        #         abs(
        #             einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
        #                 :, -1, 0, :
        #             ]
        #             - einops.rearrange(
        #                 predicted_action, "(N T) W A -> N T W A", T=obs_w
        #             )[:, -1, 0, :]
        #         )
        #     ).max()

        #     if self.sequentially_select:
        #         cbet_loss1 = self._criterion(  # F.cross_entropy
        #             cbet_logits1[:, :],
        #             action_bins[:, 0],
        #         )
        #         cbet_logits2 = self._map_to_cbet_preds_bin2(
        #             torch.cat(
        #                 (gpt_output, F.one_hot(action_bins[:, 0], num_classes=self._C)),
        #                 axis=1,
        #             )
        #         )
        #         cbet_loss2 = self._criterion(  # F.cross_entropy
        #             cbet_logits2[:, :],
        #             action_bins[:, 1],
        #         )
        #     else:
        #         cbet_loss1 = self._criterion(  # F.cross_entropy
        #             cbet_logits[:, 0, :],
        #             action_bins[:, 0],
        #         )
        #         cbet_loss2 = self._criterion(  # F.cross_entropy
        #             cbet_logits[:, 1, :],
        #             action_bins[:, 1],
        #         )
        #     cbet_loss = cbet_loss1 * 5 + cbet_loss2 * self._secondary_code_multiplier

        #     equal_total_code_rate = (
        #         torch.sum(
        #             (
        #                 torch.sum((action_bins == sampled_centers).int(), axis=1) == G
        #             ).int()
        #         )
        #         / NT
        #     )
        #     equal_single_code_rate = torch.sum(
        #         (action_bins[:, 0] == sampled_centers[:, 0]).int()
        #     ) / (NT)
        #     equal_single_code_rate2 = torch.sum(
        #         (action_bins[:, 1] == sampled_centers[:, 1]).int()
        #     ) / (NT)

        #     loss = cbet_loss + self._offset_loss_multiplier * offset_loss
        #     loss_dict = {
        #         "classification_loss": cbet_loss.detach().cpu().item(),
        #         "offset_loss": offset_loss.detach().cpu().item(),
        #         "total_loss": loss.detach().cpu().item(),
        #         "equal_total_code_rate": equal_total_code_rate,
        #         "equal_single_code_rate": equal_single_code_rate,
        #         "equal_single_code_rate2": equal_single_code_rate2,
        #         "action_diff": action_diff.detach().cpu().item(),
        #         "action_diff_tot": action_diff_tot.detach().cpu().item(),
        #         "action_diff_mean_res1": action_diff_mean_res1.detach().cpu().item(),
        #         "action_diff_mean_res2": action_diff_mean_res2.detach().cpu().item(),
        #         "action_diff_max": action_diff_max.detach().cpu().item(),
        #     }
        #     return predicted_action, loss, loss_dict

        return predicted_action, None, {}

    def preprocess_input(self, data, train_mode=True):
        for key in self.image_encoders:
            x = TensorUtils.to_float(data['obs'][key])
            x = x / 255.
            x = torch.clip(x, 0, 1)
            data['obs'][key] = x
        if train_mode:  # apply augmentation
            if self.use_augmentation:
                img_tuple = self._get_img_tuple(data)
                aug_out = self._get_aug_output_dict(self.image_aug(img_tuple))
                for img_name in self.image_encoders.keys():
                    data["obs"][img_name] = aug_out[img_name]
            return data
        return data
    
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

    def obs_encode(self, data):
        ### 1. encode image
        encoded = []
        for img_name in self.image_encoders.keys():
            x = data["obs"][img_name]
            
            B, T, C, H, W = x.shape
            e = self.image_encoders[img_name](
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

    def get_optimizers(self):
        if self.stage == 0:
            decay, no_decay = TensorUtils.separate_no_decay(self.autoencoder)
            optimizers = [
                self.optimizer_factory(params=decay),
                self.optimizer_factory(params=no_decay, weight_decay=0.)
            ]
            return optimizers
        elif self.stage == 1:
            decay, no_decay = TensorUtils.separate_no_decay(self, 
                                                            name_blacklist=('autoencoder',))
            optimizers = [
                self.optimizer_factory(params=decay),
                self.optimizer_factory(params=no_decay, weight_decay=0.)
            ]
            return optimizers
        elif self.stage == 2:
            decay, no_decay = TensorUtils.separate_no_decay(self, 
                                                            name_blacklist=('autoencoder',))
            optimizers = [
                self.optimizer_factory(params=decay),
                self.optimizer_factory(params=no_decay, weight_decay=0.)
            ]
            return optimizers

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
    
    def sample_actions(self, data):
        data = self.preprocess_input(data)

        context = self.obs_encode(data)
        predicted_act, _, _, _ = self._predict(context)

        predicted_act = einops.rearrange(predicted_act, "(N T) W A -> N T W A", T=self.obs_window_size)[:, -1, :, :]
        predicted_act = predicted_act.permute(1,0,2)
        return predicted_act.detach().cpu().numpy()

    # def get_optimizers(self):
    #     optimizer1 = self.policy_prior.configure_optimizers(
    #         weight_decay=self.optimizer_config.weight_decay,
    #         learning_rate=self.optimizer_config.lr,
    #         betas=self.optimizer_config.betas,
    #     )

    #     if self.sequentially_select:
    #         optimizer1.add_param_group(
    #             {"params": self._map_to_cbet_preds_bin1.parameters()}
    #         )
    #         optimizer1.add_param_group(
    #             {"params": self._map_to_cbet_preds_bin2.parameters()}
    #         )
    #     else:
    #         optimizer1.add_param_group(
    #             {"params": self._map_to_cbet_preds_bin.parameters()}
    #         )
    #     optimizer2 = torch.optim.AdamW(
    #         self._map_to_cbet_preds_offset.parameters(),
    #         lr=self.optimizer_config.lr,
    #         weight_decay=self.optimizer_config.weight_decay,
    #         betas=self.optimizer_config.betas,
    #     )
    #     optimizer1.add_param_group(
    #         {"params": self.image_encoders.parameters()}
    #     )
    #     # if self.visual_input:
    #     #     optimizer1.add_param_group({"params": self._resnet_header.parameters()})
    #     # self.optimizer1 = optimizer1
    #     # self.optimizer2 = optimizer2
    #     return [optimizer1, optimizer2]
    
    def get_schedulers(self, optimizers):
        return []

    # def save_model(self, path: Path):
    #     torch.save(self.state_dict(), path / "cbet_model.pt")
    #     torch.save(self.policy_prior.state_dict(), path / "gpt_model.pt")
    #     if hasattr(self, "resnet"):
    #         torch.save(self.resnet.state_dict(), path / "resnet.pt")
    #         torch.save(self._resnet_header.state_dict(), path / "resnet_header.pt")
    #     torch.save(self.optimizer1.state_dict(), path / "optimizer1.pt")
    #     torch.save(self.optimizer2.state_dict(), path / "optimizer2.pt")

    # def load_model(self, path: Path):
    #     if (path / "cbet_model.pt").exists():
    #         self.load_state_dict(torch.load(path / "cbet_model.pt"))
    #     elif (path / "gpt_model.pt").exists():
    #         self.policy_prior.load_state_dict(torch.load(path / "gpt_model.pt"))
    #     elif (path / "resnet.pt").exists():
    #         self.resnet.load_state_dict(torch.load(path / "resnet.pt"))
    #     elif (path / "optimizer1.pt").exists():
    #         self.optimizer1.load_state_dict(torch.load(path / "optimizer1.pt"))
    #     elif (path / "optimizer2.pt").exists():
    #         self.optimizer2.load_state_dict(torch.load(path / "optimizer2.pt"))
    #     else:
    #         logging.warning("No model found at %s", path)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 0, size_average: bool = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
