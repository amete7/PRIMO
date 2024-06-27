import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import time
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from quest.algos.baseline_modules.prise_utils.quantizer import VectorQuantizer
from quest.algos.baseline_modules.prise_utils.policy_head import GMMHead
# from quest.algos.baseline_modules.prise_utils.data_augmentation import BatchWiseImgColorJitterAug, TranslationAug, DataAugGroup
import quest.algos.baseline_modules.prise_utils.misc as utils



class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, input_size, inv_freq_factor=10, factor_ratio=None):
        super().__init__()
        self.input_size = input_size
        self.inv_freq_factor = inv_freq_factor
        channels = self.input_size
        channels = int(np.ceil(channels / 2) * 2)

        inv_freq = 1.0 / (
            self.inv_freq_factor ** (torch.arange(0, channels, 2).float() / channels)
        )
        self.channels = channels
        self.register_buffer("inv_freq", inv_freq)

        if factor_ratio is None:
            self.factor = 1.0
        else:
            factor = nn.Parameter(torch.ones(1) * factor_ratio)
            self.register_parameter("factor", factor)

    def forward(self, x):
        pos_x = torch.arange(x.shape[1], device=x.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        return emb_x * self.factor

    def output_shape(self, input_shape):
        return input_shape

    def output_size(self, input_size):
        return input_size

class ActionEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, action_dim):
        super().__init__()
        self.a_embedding = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.Tanh(), 
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, feature_dim),
        )
        self.sa_embedding = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim)
        )
        self.apply(utils.weight_init)
    
    def forward(self, z, a):
        u = self.a_embedding(a)
        return self.sa_embedding(z.detach()+u)


class TokenPolicy(nn.Module):
    def __init__(self, feature_dim, hidden_dim, vocab_size):
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        ).to(self.device)
    
    def forward(self, x):
        return self.net(x)

class Autoencoder(nn.Module):
    def __init__(
            self, 
            feature_dim, 
            action_dim, 
            hidden_dim, 
            # encoder, 
            n_code, 
            device, 
            decoder_type, 
            decoder_loss_coef,
        ):
        super(Autoencoder, self).__init__()

        # self.encoder = encoder
        self.device = device
        
        self.transition = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.action_encoder = ActionEncoder(feature_dim, hidden_dim, action_dim)
        
        self.a_quantizer = VectorQuantizer(n_code, feature_dim)
        
        if decoder_type== 'deterministic':
            self.decoder = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        elif decoder_type == 'gmm':
            self.decoder =  GMMHead(feature_dim, action_dim, hidden_size=hidden_dim, loss_coef = decoder_loss_coef)
        else:
            print('Decoder type not supported!')
            raise Exception
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, dim_feedforward=256, batch_first=True)
        self.transformer_embedding  = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.proj_s = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, feature_dim))
        
        self.apply(utils.weight_init)
    

# class Encoder(nn.Module):
#     def __init__(self, obs_shape, feature_dim):
#         super().__init__()

#         assert len(obs_shape) == 3
#         repr_dim = 32 * 35 * 35

#         self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
#                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
#                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
#                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
#                                  nn.ReLU())
#         self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
#                                    nn.LayerNorm(feature_dim), nn.Tanh())
#         self.repr_dim = feature_dim
        
#         self.apply(utils.weight_init)

#     def forward(self, obs):
#         obs = obs / 255.0 - 0.5
#         h = self.convnet(obs)
#         h = h.view(h.shape[0], -1)
#         return self.trunk(h)
    