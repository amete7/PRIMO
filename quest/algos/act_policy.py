import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import torchvision.transforms as transforms
import quest.utils.tensor_utils as TensorUtils
from quest.utils.utils import map_tensor_to_device
import quest.utils.obs_utils as ObsUtils
from quest.algos.base import ChunkPolicy

class ACTPolicy(ChunkPolicy):
    def __init__(
            self, 
            act_model,
            image_encoder_factory,
            lowdim_encoder_factory,
            obs_proj,
            task_encoder,
            image_aug_factory,
            optimizer_factory,
            scheduler_factory,
            loss_fn,
            kl_weight,
            lr_backbone,
            shape_meta,
            action_horizon,
            device
            ):
        super().__init__(
            image_encoder_factory,
            lowdim_encoder_factory,
            image_aug_factory,
            task_encoder,
            obs_proj,
            shape_meta,
            action_horizon,
            device)
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.shape_meta = shape_meta
        self.loss_fn = loss_fn
        self.kl_weight = kl_weight
        self.lr_backbone = lr_backbone
        
        self.act_model = act_model.to(device)

    def forward(self, data):
        text_encoded = self.get_task_emb(data)  # (B, E)
        qpos = []
        for name, shape in self.shape_meta['observation']['lowdim'].items():
            qpos.append(data["obs"][name])
        qpos = torch.cat(qpos, -1)[:, -1, :]
        image = []
        for name, shape in self.shape_meta["observation"]['rgb'].items():
            image.append(data["obs"][name])
        image = torch.cat(image, 1)
        normalize_image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize_image(image)
        if 'actions' in data:
            is_pad = torch.zeros((data["actions"].shape[0], data["actions"].shape[1]), device=self.device, dtype=torch.bool)
            pred_action, _, latent = self.act_model(
                qpos, image, None, text_encoded, data["actions"], is_pad
            )
        else:
            pred_action, _, latent = self.act_model(
                qpos, image, None, text_encoded
            )
        return pred_action, latent

    def compute_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        pred_action, latent = self.forward(data)
        l1_loss = self.loss_fn(pred_action, data["actions"])
        total_kld, dim_wise_kld, mean_kld = kl_divergence(latent[0], latent[1])
        loss = l1_loss + total_kld[0]*self.kl_weight
        info = {
            'l1_loss': l1_loss.item(),
            'total_kld': total_kld[0].item(),
            'mean_kld': mean_kld.item(),
            'total_loss': loss.item(),
        }
        return loss, info
    
    def sample_actions(self, data):
        data = self.preprocess_input(data, train_mode=False)
        pred_action, _ = self.forward(data)
        pred_action = pred_action.permute(1, 0, 2)
        return pred_action.detach().cpu().numpy()


    def get_optimizers(self):
        decay, no_decay = TensorUtils.separate_no_decay(self, 
                                                        name_blacklist=('backbones',))
        backbone_decay, backbone_no_decay = TensorUtils.separate_no_decay(self.act_model.backbones)
        optimizers = [
            self.optimizer_factory(params=itertools.chain(decay, backbone_decay)),
            self.optimizer_factory(params=itertools.chain(no_decay, backbone_no_decay), weight_decay=0.)
        ]
        return optimizers

    def get_schedulers(self, optimizers):
        return [self.scheduler_factory(optimizer=optimizer) for optimizer in optimizers]

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld