'''
This code is adapted from https://github.dev/Toni-SM/skrl/blob/main/skrl/agents/torch/ppo/ppo.py 
'''

from typing import Any, Mapping, Optional, Tuple, Union

import copy
import itertools
import gymnasium

import torch
import torch.nn as nn
import torch.nn.functional as F
from quest.algos.utils.ppo_modules import KLAdaptiveLR
import wandb

class PPO(nn.Module):
    def __init__(self,
                 ppo_model,
                 memory,
                 observation_space,
                 spt_kldiv_scale,
                 update_interval,
                 learning_starts_steps,
                 discount_factor,
                 lambda_adv,
                 mini_batches,
                 learning_epochs,
                 kl_threshold,
                 entropy_loss_scale,
                 value_loss_scale,
                 grad_norm_clip,
                 clip_predicted_values,
                 value_clip,
                 ratio_clip,
                 use_lr_scheduler,
                 num_envs,
                 device,
                 use_amp,
                 ):
        # models
        self.ppo_model = ppo_model
        self.spt_model = copy.deepcopy(ppo_model)
        self.memory = memory
        self.observation_space = observation_space
        self.device = device
        self.block_size = self.ppo_model.block_size
        self._spt_kldiv_scale = spt_kldiv_scale
        self.update_interval = update_interval
        self._learning_starts = learning_starts_steps
        self._discount_factor = discount_factor
        self._lambda = lambda_adv
        self._mini_batches = mini_batches
        self._learning_epochs = learning_epochs
        self._kl_threshold = kl_threshold
        self._entropy_loss_scale = entropy_loss_scale
        self._value_loss_scale = value_loss_scale
        self._grad_norm_clip = grad_norm_clip
        self._clip_predicted_values = clip_predicted_values
        self._value_clip = value_clip
        self._ratio_clip = ratio_clip
        self.num_envs = num_envs
        self._learning_rate_scheduler = use_lr_scheduler
        self._rollout = 0
        self._train_step = 0

        # optimizer
        self.optimizers = self.ppo_model.get_optimizers()
        if self._learning_rate_scheduler:
            self.schedulers = self.ppo_model.get_schedulers(self.optimizers)
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    def init(self):
        """Initialize the agent
        """
        self.eval()
        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="obs", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="indices", size=self.block_size, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=self.block_size, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=self.block_size, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=self.block_size, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=self.block_size, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=self.block_size, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=self.block_size, dtype=torch.float32)

            # tensors sampled during training
            self._tensors_names = ["obs", "indices", "log_prob", "values", "returns", "advantages"]

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_obs = None
    
    def act(self, obs):
        """Process the environment's states to make a decision (actions) using the main policy
        """
        # sample stochastic actions
        env_actions, indices, log_prob, values = self.ppo_model.get_action(obs)
        self._current_log_prob = log_prob

        return env_actions, indices, log_prob, values

    def record_transition(self,
                          obs,
                          indices,
                          env_rewards,
                          values,
                          next_obs,
                          terminated,
                          truncated,
                          ):
        """Record an environment transition in memory
        """
        if self.memory is not None:
            self._current_next_obs = next_obs
            # compute rewards
            spt_log_prob, _, _ = self.spt_model.get_log_prob_with_values(obs, indices)
            rewards = -self._spt_kldiv_scale * (self._current_log_prob - spt_log_prob)
            rewards[:,-1] = env_rewards
            # storage transition in memory
            self.memory.add_samples(obs=obs, indices=indices, rewards=rewards, next_obs=next_obs,
                                    terminated=terminated, truncated=truncated, log_prob=self._current_log_prob, values=values)

    def post_interaction(self, timestep):
        """Callback called after the interaction with the environment
        """
        self._rollout += 1
        if not self._rollout % self.update_interval and timestep >= self._learning_starts:
            self.train()
            self._update()
            self.eval()
    
    def _update(self):
        """Algorithm's main update step
        """
        def compute_gae(rewards: torch.Tensor,
                        dones: torch.Tensor,
                        values: torch.Tensor,
                        last_values: torch.Tensor,
                        discount_factor: float = 0.99,
                        lambda_coefficient: float = 0.95) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)
            """
            rewards = rewards.permute(0, 2, 1).reshape(-1, self.num_envs)
            dones = dones.permute(0, 2, 1).reshape(-1, self.num_envs)
            values = values.permute(0, 2, 1).reshape(-1, self.num_envs)
            last_values = last_values.permute(1, 0)
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = rewards[i] - values[i] + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        # compute returns and advantages
        with torch.no_grad():
            self.eval()
            last_values = self.ppo_model.get_values(self._current_next_obs, None)
            self.train()
        
        returns, advantages = compute_gae(rewards=self.memory.get_tensor_by_name("rewards"),
                                          dones=self.memory.get_tensor_by_name("terminated"),
                                          values=self.memory.get_tensor_by_name("values"),
                                          last_values=last_values,
                                          discount_factor=self._discount_factor,
                                          lambda_coefficient=self._lambda)

        self.memory.set_tensor_by_name("returns", returns)
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for sampled_obs, sampled_indices, sampled_log_prob, sampled_values, sampled_returns, sampled_advantages in sampled_batches:
                # compute next log probabilities TODO: check if it's ok to compute everything in one forward pass | move to GPU
                next_log_prob, predicted_values, logits = self.ppo_model.get_log_prob_with_values(sampled_obs, sampled_indices)

                # compute approximate KL divergence
                with torch.no_grad():
                    ratio = next_log_prob - sampled_log_prob
                    kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                    kl_divergences.append(kl_divergence)

                # early stopping with KL divergence
                if self._kl_threshold and kl_divergence > self._kl_threshold:
                    break

                # compute entropy loss
                if self._entropy_loss_scale:
                    entropy_loss = -self._entropy_loss_scale * self.ppo_model.get_entropy(logits).mean()
                else:
                    entropy_loss = 0

                # compute policy loss
                ratio = torch.exp(next_log_prob - sampled_log_prob)
                surrogate = sampled_advantages * ratio
                surrogate_clipped = sampled_advantages * torch.clip(ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)

                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                # compute value loss
                if self._clip_predicted_values:
                    predicted_values = sampled_values + torch.clip(predicted_values - sampled_values,
                                                                   min=-self._value_clip,
                                                                   max=self._value_clip)
                value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                #self.optimizer.zero_grad()
                loss = policy_loss + entropy_loss + value_loss
                self.scaler.scale(loss).backward()
                for optimizer in self.optimizers:
                    self.scaler.unscale_(optimizer)
                if self._grad_norm_clip > 0:
                    grad_norm = nn.utils.clip_grad_norm_(self.ppo_model.parameters(), self._grad_norm_clip)
                # optimizer.step()
                for optimizer in self.optimizers:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
                
                # log wandb
                info = {
                    "policy_loss": policy_loss.item(),
                    "entropy_loss": entropy_loss.item(),
                    "value_loss": value_loss.item(),
                    "kl_divergence": kl_divergence.item(),
                    "grad_norm": grad_norm.item(),
                }
                wandb.log(info, step=self._train_step)
                self._train_step += 1

            # update learning rate
            if self._learning_rate_scheduler:
                for scheduler in self.schedulers:
                    if isinstance(scheduler, KLAdaptiveLR):
                        scheduler.step(torch.tensor(kl_divergences).mean())
                    else:
                        scheduler.step()