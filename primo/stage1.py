import torch
import torch.nn as nn
import primo
from primo.modules import v1

class SkillVAE_Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        policy_cfg = cfg.policy
        self.skill_vae = eval(f"{policy_cfg.model_type}.SkillVAE")(policy_cfg)
        self.using_vq = True if policy_cfg.vq_type == "vq" else False

        if cfg.train.loss_type == "mse":
            self.loss = torch.nn.MSELoss()
        elif cfg.train.loss_type == "l1":
            self.loss = torch.nn.L1Loss()
        else:
            raise NotImplementedError(f"Unknown loss type {cfg.train.loss_type}")

    def forward(self, data):
        pred, pp, pp_sample, aux_loss = self.skill_vae(data["actions"])
        info = {'pp': pp, 'pp_sample': pp_sample, 'aux_loss': aux_loss.sum()}
        return pred, info

    def compute_loss(self, data):
        pred, info = self.forward(data)
        loss = self.loss(pred, data["actions"])
        if self.using_vq:
            loss += info['aux_loss']
        return loss, info