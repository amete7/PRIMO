import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import quest.utils.tensor_utils as TensorUtils
# from quest.algos.utils.data_augmentation import *
# from quest.modules.rgb_modules.rgb_modules import ResnetEncoder
from quest.algos.utils.mlp_proj import MLPProj
from quest.algos.baseline_modules.diffusion_modules import ConditionalUnet1D
from diffusers.training_utils import EMAModel
from quest.utils.utils import map_tensor_to_device
import quest.utils.obs_utils as ObsUtils


class DiffusionPolicy(nn.Module):
    def __init__(
            self, 
            diffusion_model,
            optimizer_factory,
            scheduler_factory,
            image_encoder_factory,
            proprio_encoder,
            image_aug,
            action_horizon,
            n_tasks,
            lang_emb_dim,
            shape_meta,
            device
            ):
        super().__init__()
        self.device = device
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.action_horizon = action_horizon
        self.action_queue = None
        
        self.diffusion_model = diffusion_model.to(device)

        self.task_encodings = nn.Embedding(n_tasks, lang_emb_dim)

        # observation encoders
        image_encoders = {}
        for name in shape_meta["image_inputs"]:
            image_encoders[name] = image_encoder_factory()
        self.image_encoders = nn.ModuleDict(image_encoders)
        self.proprio_encoder = proprio_encoder

        # add data augmentation for rgb inputs
        self.image_aug = image_aug

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
        lang_emb = self.task_encodings(data["task_id"])
        cond = torch.cat([init_obs, lang_emb], dim=-1)
        loss = self.diffusion_model(cond,data["actions"])
        return loss

    def compute_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        loss = self.forward(data)
        return loss, {}
    
    def get_action(self, obs, task_id):
        assert self.action_queue is not None, "you need to call quest.reset() before getting actions"

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
        data = self.preprocess_input(data, train_mode=False)
        init_obs = self.obs_encode(data)
        lang_emb = self.task_encodings(data["task_id"])
        cond = torch.cat([init_obs, lang_emb], dim=-1)
        actions = self.diffusion_model.get_action(cond)
        actions = actions.permute(1,0,2)
        return actions.detach().cpu().numpy()

    def get_optimizers(self):
        decay, no_decay = TensorUtils.separate_no_decay(self)
        optimizers = [
            self.optimizer_factory(params=decay),
            self.optimizer_factory(params=no_decay, weight_decay=0.)
        ]
        return optimizers
            
    def get_schedulers(self, optimizers):
        return [self.scheduler_factory(optimizer=optimizer) for optimizer in optimizers]

    def reset(self):
        self.action_queue = deque(maxlen=self.action_horizon)
        
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


class DiffusionModel(nn.Module):
    def __init__(self, 
                 noise_scheduler,
                 action_dim,
                 global_cond_dim,
                 diffusion_step_emb_dim,
                 down_dims,
                 ema_power,
                 device):
        super().__init__()
        self.device = device
        net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_emb_dim,
            down_dims=down_dims,
        ).to(self.device)
        self.ema = EMAModel(
            model=net,
            power=ema_power)
        self.net = net
        self.noise_scheduler = noise_scheduler

    def forward(self, cond, actions):
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (cond.shape[0],), device=self.device
        ).long()
        noise = torch.randn(actions.shape, device=self.device)
        # add noise to the clean actions according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = self.noise_scheduler.add_noise(
            actions, noise, timesteps)
        # predict the noise residual
        noise_pred = self.net(
            noisy_actions, timesteps, global_cond=cond)
        loss = F.mse_loss(noise_pred, noise)
        return loss

    def get_action(self, cond):
        nets = self.net
        noisy_action = torch.randn(
            (cond.shape[0], self.skill_block_size, self.action_dim), device=self.device)
        naction = noisy_action
        # init scheduler
        self.noise_scheduler.set_timesteps(self.diffusion_inf_steps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets(
                sample=naction, 
                timestep=k,
                global_cond=cond
            )
            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample
        return naction

    def ema_update(self):
        self.ema.step(self.net)