from torch import nn
import torch
from torch.nn import functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from quest.algos.utils.mlp_proj import MLPProj
from collections import deque
import quest.utils.tensor_utils as TensorUtils
from quest.algos.utils.data_augmentation import *
import quest.utils.utils as utils

class PPO_Model(nn.Module):
    def __init__(self,
                 autoencoder,
                 network_body,
                 policy_head,
                 value_head,
                 image_encoder_factory,
                 proprio_encoder,
                 image_aug,
                 optimizer_factory,
                 scheduler_factory,
                 start_token,
                 block_size,
                 beam_size,
                 temperature,
                 n_tasks,
                 cat_obs_dim,
                 action_horizon,
                 shape_meta,
                 device):
        super().__init__()
        self.autoencoder = autoencoder
        self.body = network_body
        self.policy_head = policy_head
        self.value_head = value_head
        self.start_token = start_token
        self.block_size = block_size
        self.beam_size = beam_size
        self.temperature = temperature
        self.action_horizon = action_horizon
        self.use_augmentation = image_aug is not None
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.device = device
        self.task_id = None
        self.single_env_space = None # TODO: check if can be initialized via hydra

        self.task_encodings = nn.Embedding(n_tasks, self.body.n_embd)
        self.obs_proj = MLPProj(cat_obs_dim, self.body.n_embd)

        # observation encoders
        image_encoders = {}
        for name in shape_meta["image_inputs"]:
            image_encoders[name] = image_encoder_factory()
        self.image_encoders = nn.ModuleDict(image_encoders)
        self.proprio_encoder = proprio_encoder

        # add data augmentation for rgb inputs
        self.image_aug = image_aug
    
    def get_optimizers(self):
        decay, no_decay = TensorUtils.separate_no_decay(self, 
                                                            name_blacklist=(
                                                                'autoencoder',))
        optimizers = [
            self.optimizer_factory(params=decay),
            self.optimizer_factory(params=no_decay, weight_decay=0.)
        ]
        return optimizers

    def get_schedulers(self, optimizers):
        return [self.scheduler_factory(optimizer=optimizer) for optimizer in optimizers]
    
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
        encoded.append(self.proprio_encoder(data["obs"]['robot_state']))  # add (B, T, H_extra)
        encoded = torch.cat(encoded, -1)  # (B, T, H_all)
        init_obs_emb = self.obs_proj(encoded)
        task_emb = self.task_encodings(data["task_id"]).unsqueeze(1).repeat(init_obs_emb.shape[0], 1, 1)
        context = torch.cat([task_emb, init_obs_emb], dim=1)
        return context

    def forward(self, idx, context):
        body_output = self.body(idx, context)
        policy_output = self.policy_head(body_output)
        value_output = self.value_head(body_output)
        return policy_output, value_output

    # def reset(self):
    #     self.action_queue = deque(maxlen=self.action_horizon)
    def init(self, single_env_space, task_id):
        self.single_env_space = single_env_space
        self.task_id = task_id
    
    def preprocess_input(self, data, train_mode=True):
        data = TensorUtils.recursive_dict_list_tuple_apply(
                data, {torch.Tensor: lambda x: x.unsqueeze(dim=1)})  # add time dimension
        data["task_id"] = data["task_id"].squeeze(1)
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

    def get_action(self, obs):#TODO: check if train_mode arg should be passed
        data = self.preprocess_input(self.get_data(obs), train_mode=False)
        context = self.obs_encode(data)
        idx = torch.ones((context.shape[0], 1), dtype=torch.long, device=self.device) * self.start_token
        log_probs = []
        values = []
        for i in range(self.block_size):
            logits, value = self.forward(idx, context)
            next_indices, log_prob = top_k_sampling(logits[:,-1,:], self.beam_size, self.temperature)
            idx = torch.cat([idx, next_indices], dim=1)
            log_probs.append(log_prob)
            values.append(value[:,-1,:])
        indices = idx[:,1:]
        actions = self.autoencoder.decode_actions(indices).detach().cpu().numpy()
        return actions[:,:self.action_horizon,:], indices, torch.cat(log_probs, dim=1), torch.cat(values, dim=1)
    
    def get_values(self, obs, indices):
        data = self.preprocess_input(self.get_data(obs), train_mode=False)
        context = self.obs_encode(data)
        if indices is not None:
            indices = torch.cat([torch.ones((context.shape[0], 1), dtype=torch.long, device=self.device) * self.start_token, indices[..., :-1]], dim=1)
        else:
            indices = torch.ones((context.shape[0], 1), dtype=torch.long, device=self.device) * self.start_token
        values = self.value_head(self.body(indices, context))
        return values.squeeze(-1)
    
    def get_log_prob_with_values(self, obs, indices):#TODO: check if train_mode arg should be passed
        data = self.preprocess_input(self.get_data(obs), train_mode=False)
        context = self.obs_encode(data)
        indices_in = torch.cat([torch.ones((context.shape[0], 1), dtype=torch.long, device=self.device) * self.start_token, indices[..., :-1]], dim=1)
        logits, values = self.forward(indices_in, context)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
        return log_probs, values.squeeze(-1), logits
    
    def get_entropy(self, logits):#TODO: reconfirm if this is the correct implementation
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(-1)
        return entropy

    def get_data(self, obs):
        obs = TensorUtils.tensor_to_space(obs, self.single_env_space)
        data = {
            "obs": obs,
            "task_id": torch.tensor(self.task_id, dtype=torch.long, device=self.device).unsqueeze(0)
        }
        return data

class SkillGPT_Body(nn.Module):
    def __init__(self,
                 vocab_size,
                 n_layer,
                 n_head,
                 n_embd,
                 attn_pdrop,
                 embd_pdrop,
                 ):
        super().__init__()
        self.n_embd = n_embd
        self.tok_emb = nn.Embedding(vocab_size+1, n_embd)
        self.add_positional_emb = Summer(PositionalEncoding1D(n_embd))
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4*n_embd,
                dropout=attn_pdrop,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ),
            num_layers=n_layer
        )
        self.drop = nn.Dropout(embd_pdrop)
        self.lnf = nn.LayerNorm(n_embd)

    def forward(self, idx, context):
        x = self.tok_emb(idx)
        x = self.add_positional_emb(x)
        x = torch.cat([context, x], dim=1)
        x = self.drop(x)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1),x.device)
        x = self.decoder(x, mask=mask, is_causal=True)
        x = x[:, context.size(1):, :]
        x = self.lnf(x)
        return x

class PolicyHead(nn.Module):
    def __init__(self, 
                 n_embd, 
                 vocab_size, 
                 ):
        super().__init__()
        self.head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x):
        return self.head(x)

class ValueHead(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.head = nn.Linear(n_embd, 1)
        nn.init.orthogonal_(self.head.weight, gain=1)
        nn.init.constant_(self.head.bias, 0)
        
    def forward(self, x):
        return self.head(x)

def top_k_sampling(logits, k, temperature=1.0):
    scaled_logits = logits / temperature
    top_values, top_indices = torch.topk(scaled_logits, k, dim=-1)
    top_probs = torch.softmax(top_values, dim=-1)
    sampled_indices = torch.multinomial(top_probs, num_samples=1, replacement=True)
    original_indices = top_indices.gather(-1, sampled_indices)
    log_probs = F.log_softmax(logits, dim=-1)
    log_prob = log_probs.gather(-1, original_indices)
    return original_indices, log_prob


