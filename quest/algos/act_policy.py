import torch
import torch.nn as nn
import torch.nn.functional as F
import quest.utils.tensor_utils as TensorUtils
from quest.utils.utils import map_tensor_to_device
import quest.utils.obs_utils as ObsUtils
from quest.algos.base import ChunkPolicy

class ACTPolicy(ChunkPolicy):
    def __init__(
            self, 
            act_model,
            optimizer_factory,
            scheduler_factory,
            image_encoder_factory,
            proprio_encoder,
            obs_proj,
            task_encoder,
            image_aug,
            shape_meta,
            device
            ):
        super().__init__(
            image_encoder_factory, 
            proprio_encoder, 
            obs_proj, 
            image_aug, 
            shape_meta, 
            device)
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.task_encoder = task_encoder
        
        self.act_model = act_model.to(device)

    def temporal_encode(self, x):
        pos_emb = self.temporal_position_encoding_fn(x)
        x = x + pos_emb.unsqueeze(1)  # (B, T, num_modality, E)
        sh = x.shape
        self.temporal_transformer.compute_mask(x.shape)

        x = TensorUtils.join_dimensions(x, 1, 2)  # (B, T*num_modality, E)
        x = self.temporal_transformer(x)
        x = x.reshape(*sh)
        return x[:, :, 0]  # (B, T, E)

    def spatial_encode(self, data):
        # 1. encode proprio
        extra = self.proprio_encoder(data["obs"]['robot_states']).unsqueeze(2)
        
        # 2. encode language, treat it as a seperate token
        B, T = extra.shape[:2]
        text_encoded = self.task_encoder(data["task_id"])  # (B, E)
        text_encoded = text_encoded.view(B, 1, 1, -1).expand(
            -1, T, -1, -1
        )  # (B, T, 1, E)
        encoded = [text_encoded, extra]

        # 3. encode image
        for img_name in self.image_encoders.keys():
            x = data["obs"][img_name]
            B, T, C, H, W = x.shape
            img_encoded = self.image_encoders[img_name](
                x.reshape(B * T, C, H, W),
                ).view(B, T, 1, -1)
            encoded.append(img_encoded)
        encoded = torch.cat(encoded, -2)  # (B, T, num_modalities, E)
        return encoded

    def forward(self, data):
        text_encoded = self.task_encoder(data["task_id"])  # (B, E)
        

    def compute_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        dist = self.forward(data)
        loss = self.policy_head.loss_fn(dist, data["actions"], self.reduction)
        info = {
            'loss': loss.item(),
        }
        return loss, info
    
    def get_action(self, obs, task_id):
        self.eval()
        for key, value in obs.items():
            if key in self.image_encoders:
                value = ObsUtils.process_frame(value, channel_dim=3)
            obs[key] = torch.tensor(value).unsqueeze(0)
        batch = {}
        batch["obs"] = obs
        batch["task_id"] = torch.tensor([task_id], dtype=torch.long)
        batch = map_tensor_to_device(batch, self.device)
        batch = self.preprocess_input(batch, train_mode=False)
        with torch.no_grad():
            x = self.spatial_encode(batch)
            x = self.temporal_encode(x)
            dist = self.policy_head(x[:, -1])
        action = dist.sample().squeeze().cpu().numpy()
        return action


    def get_optimizers(self):
        decay, no_decay = TensorUtils.separate_no_decay(self)
        optimizers = [
            self.optimizer_factory(params=decay),
            self.optimizer_factory(params=no_decay, weight_decay=0.)
        ]
        return optimizers

    def get_schedulers(self, optimizers):
        return [self.scheduler_factory(optimizer=optimizer) for optimizer in optimizers]
