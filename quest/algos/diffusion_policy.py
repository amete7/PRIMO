import torch
import torch.nn as nn
from collections import deque
import quest.utils.tensor_utils as TensorUtils
# from quest.algos.utils.data_augmentation import *
# from quest.modules.rgb_modules.rgb_modules import ResnetEncoder
from quest.algos.utils.mlp_proj import MLPProj
from quest.algos.baseline_modules.diffusion_modules import ConditionalUnet1D
from diffusers.training_utils import EMAModel


class DiffusionPolicy(nn.Module):
    def __init__(
            self, 
            diffusion_model,

            shape_meta,
            device
            ):
        super().__init__()
        # policy_cfg = cfg.policy
        self.device = cfg.device
        self.use_augmentation = cfg.train.use_augmentation
        self.mpc_horizon = policy_cfg.mpc_horizon
        self.action_queue = deque(maxlen=self.mpc_horizon)
        
        self.diff_model = DiffusionModel(policy_cfg, self.device)
        self.diff_model = self.diff_model.to(self.device)
        # self.input_proj = MLPProj(policy_cfg.cat_obs_dim+policy_cfg.lang_emb_dim, policy_cfg.cond_dim)

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

        self.net = ConditionalUnet1D(
            input_dim=cfg.action_dim,
            global_cond_dim=cfg.global_cond_dim,
            diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
            down_dims=cfg.down_dims,
        ).to(self.device)
        self.proprio_encoder = MLPProj(shape_meta["all_shapes"]['robot_states'][0], policy_cfg.proprio_emb_dim)
        self.task_encodings = nn.Embedding(cfg.n_tasks, policy_cfg.lang_emb_dim)

            # add data augmentation for rgb inputs
        color_aug = eval(policy_cfg.color_aug.network)(**policy_cfg.color_aug.network_kwargs)
        policy_cfg.translation_aug.network_kwargs["input_shape"] = shape_meta["all_shapes"][cfg.data.obs.modality.rgb[0]]
        translation_aug = eval(policy_cfg.translation_aug.network)(**policy_cfg.translation_aug.network_kwargs)
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
        lang_emb = self.task_encodings(data["task_id"])
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
        lang_emb = self.task_encodings(data["task_id"])
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
        # if cfg.scheduler == 'ddpm':
        #     self.noise_scheduler = DDPMScheduler(
        #         num_train_timesteps=cfg.diffusion_train_steps,
        #         beta_schedule='squaredcos_cap_v2',
        #     )
        # elif cfg.scheduler == 'ddim':
        #     self.noise_scheduler = DDIMScheduler(
        #         num_train_timesteps=cfg.diffusion_train_steps,
        #         beta_schedule='squaredcos_cap_v2',
        #     )
        # else:
        #     raise NotImplementedError(f'Invalid scheduler type: {cfg.scheduler}')

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
            (cond.shape[0], self.cfg.skill_block_size, self.cfg.action_dim), device=self.device)
        naction = noisy_action
        # init scheduler
        self.noise_scheduler.set_timesteps(self.cfg.diffusion_inf_steps)

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