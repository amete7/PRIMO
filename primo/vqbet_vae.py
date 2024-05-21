import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn

from primo.vqbet_modules.vqvae import VqVae

class VQVAE_Model(nn.Module):
    def __init__(self, cfg):
        super().__init__(cfg)
        policy_cfg = cfg.policy
        self.vq_vae = VqVae(
            input_dim_h=policy_cfg.skill_block_size,  # length of action chunk
            input_dim_w=policy_cfg.action_dim,  # action dim
            n_latent_dims=policy_cfg.codebook_dim,
            vqvae_n_embed=policy_cfg.codebook_size,
            vqvae_groups=policy_cfg.num_codebooks,
            hidden_dim=policy_cfg.hidden_dim,
            num_layers=policy_cfg.num_layers,
            device=self.device,
        )

    def forward(self, data):
        pred, total_loss, l1_loss, codebook_loss, pp = self.vq_vae(data["actions"])
        info = {'recon_loss': l1_loss, 'codebook_loss': codebook_loss, 'pp': pp}
        return pred, total_loss, info

    def compute_loss(self, data):
        pred, total_loss, info = self.forward(data)
        return total_loss, info