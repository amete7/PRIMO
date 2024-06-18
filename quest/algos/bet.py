import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import deque

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from quest.algos.baseline_modules.vq_behavior_transformer.utils import MLP
import quest.utils.tensor_utils as TensorUtils


from quest.algos.base import ChunkPolicy

class BehaviorTransformer(ChunkPolicy):
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
        obs_proj,
        task_encoder,
        image_aug,
        loss_fn,
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
        super().__init__(
            image_encoder_factory, 
            proprio_encoder, 
            obs_proj, 
            image_aug, 
            shape_meta, 
            action_horizon,
            device)
        # self._obs_dim = obs_dim

        self.autoencoder = autoencoder
        self.policy_prior = policy_prior
        self.stage = stage
        self.optimizer_config = optimizer_config
        self.optimizer_factory = optimizer_factory
        self.task_encoder = task_encoder

        self.obs_window_size = obs_window_size
        self.skill_block_size = skill_block_size
        self.sequentially_select = sequentially_select
        # if goal_dim <= 0:
        #     self._cbet_method = self.GOAL_SPEC.unconditional
        # elif obs_dim == goal_dim:
        
        # TODO maybe unhardcode this (maybe)
        self._cbet_method = self.GOAL_SPEC.concat
        # else:
        #     self._cbet_method = self.GOAL_SPEC.stack

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

    def compute_loss(self, data):
        if self.stage == 0:
            return self.compute_autoencoder_loss(data)
        elif self.stage == 1:
            return self.compute_prior_loss(data)
        elif self.stage == 2:
            return self.compute_prior_loss(data)

    def compute_autoencoder_loss(self, data):
        pred, total_loss, l1_loss, codebook_loss, pp = self.autoencoder(data["actions"])
        info = {
            'recon_loss': l1_loss.item(), 
            'codebook_loss': codebook_loss.item(), 
            'pp': pp}
        return total_loss, info
    
    def compute_prior_loss(self, data):
        data = self.preprocess_input(data)

        breakpoint()

        context = self.get_context(data)
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
            "classification_loss": cbet_loss.detach().cpu().item(),
            "offset_loss": offset_loss.detach().cpu().item(),
            "total_loss": loss.detach().cpu().item(),
            "equal_total_code_rate": equal_total_code_rate.item(),
            "equal_single_code_rate": equal_single_code_rate.item(),
            "equal_single_code_rate2": equal_single_code_rate2.item(),
            "action_diff": action_diff.detach().cpu().item(),
            "action_diff_tot": action_diff_tot.detach().cpu().item(),
            "action_diff_mean_res1": action_diff_mean_res1.detach().cpu().item(),
            "action_diff_mean_res2": action_diff_mean_res2.detach().cpu().item(),
            "action_diff_max": action_diff_max.detach().cpu().item(),
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

        # if self._cbet_method == self.GOAL_SPEC.unconditional:
        #     gpt_output = gpt_output
        # else:
        #     gpt_output = gpt_output[:, goal_seq.size(1) :, :]

        # TODO: this might cause some bugs
        breakpoint()
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
    
    def sample_actions(self, data):
        data = self.preprocess_input(data)

        context = self.get_context(data)
        predicted_act, _, _, _ = self._predict(context)

        predicted_act = einops.rearrange(predicted_act, "(N T) W A -> N T W A", T=self.obs_window_size)[:, -1, :, :]
        predicted_act = predicted_act.permute(1,0,2)
        return predicted_act.detach().cpu().numpy()

    def get_context(self, data):
        obs_emb = self.obs_encode(data)
        task_emb = self.task_encoder(data["task_id"]).unsqueeze(1)
        context = torch.cat([task_emb, obs_emb], dim=1)
        return context


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
