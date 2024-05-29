import math
import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from vector_quantize_pytorch import VectorQuantize, FSQ
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer



###############################################################################
#
# Positional Embedding module
#
###############################################################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:,:x.size(1), :]
        return self.dropout(pe.repeat(x.size(0),1,1))

###############################################################################
#
# MLP projection module
#
###############################################################################


class MLP_Proj(nn.Module):
    """
    Encode any embedding

    h = f(e), where
        e: embedding from some model
        h: latent embedding (B, H)
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super().__init__()
        assert num_layers >= 1, "[error] num_layers < 1"
        sizes = [input_size] + [hidden_size] * (num_layers - 1) + [output_size]
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.projection = nn.Sequential(*layers)

    def forward(self, data):
        """
        data:
            task_emb: (B, E)
        """
        h = self.projection(data)  # (B, H)
        return h

###############################################################################
#
# obs encoder modules
#
###############################################################################

class ObsEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoders = nn.ModuleList()
        for i in range(len(cfg.input_dim)):
            encoder = MLP_Proj(cfg.input_dim[i], cfg.output_dim[i], cfg.output_dim[i], num_layers=cfg.num_layers[i], dropout=cfg.dropout[i])
            self.encoders.append(encoder)
        self.encoders.append(MLP_Proj(sum(cfg.output_dim), cfg.proj_dim, cfg.proj_dim, num_layers=1))
    
    def forward(self, obs):
        x = [encoder(obs[i]) for i, encoder in enumerate(self.encoders[:-1])]
        x = torch.cat(x, dim=-1)
        x = self.encoders[-1](x)
        return x.unsqueeze(1)

###############################################################################
#
# 1D conv modules
#
###############################################################################

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride, no_pad=False):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if no_pad:
            self.padding = 0
        else:
            self.padding = dilation*(kernel_size-1)
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, stride=stride)

    def forward(self, x):
        x = self.conv(x)
        last_n = (2*self.padding-self.kernel_size)//self.stride + 1
        if last_n> 0:
            return x[:, :, :-last_n]
        else:
            return x


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
        from https://github.com/jannerm/diffuser/blob/06b8e6a042e6a3312d50ed8048cba14afeab3085/diffuser/models/helpers.py#L46
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, stride, n_groups=4, causal=True, no_pad=False):
        super().__init__()
        if causal:
            conv = CausalConv1d(inp_channels, out_channels, kernel_size, dilation=1, stride=stride, no_pad=no_pad)
        else:
            conv = nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride)

        self.block = nn.Sequential(
            conv,
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )
    def forward(self, x):
        return self.block(x)

class CausalDeConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride):
        super(CausalDeConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.conv(x)
        last_n = self.kernel_size-self.stride
        if last_n> 0:
            return x[:, :, :-last_n]
        else:
            return x

class DeConv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
        from https://github.com/jannerm/diffuser/blob/06b8e6a042e6a3312d50ed8048cba14afeab3085/diffuser/models/helpers.py#L46
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, stride, n_groups=8, causal=True):
        super().__init__()
        if causal:
            conv = CausalDeConv1d(inp_channels, out_channels, kernel_size, dilation=1, stride=stride)
        else:
            conv = nn.ConvTranspose1d(inp_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride, output_padding=stride-1)

        self.block = nn.Sequential(
            conv,
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size=[5,3], stride=[2,2], n_groups=8, causal=True, residual=False, pooling_layers=[]):
        super().__init__()
        self.pooling_layers = pooling_layers
        self.blocks = nn.ModuleList()
        for i in range(len(kernel_size)):
            block = Conv1dBlock(
                inp_channels if i == 0 else out_channels, 
                out_channels, 
                kernel_size[i], 
                stride[i], 
                n_groups=n_groups, 
                causal=causal
            )
            self.blocks.append(block)
        if residual:
            if out_channels == inp_channels and stride[0] == 1:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.Conv1d(inp_channels, out_channels, kernel_size=1, stride=sum(stride))
        if pooling_layers:
            self.pooling = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, input_dict):
        x = input_dict
        x = torch.transpose(x, 1, 2)
        out = x
        layer_num = 0
        for block in self.blocks:
            out = block(out)
            if hasattr(self, 'pooling'):
                if layer_num in self.pooling_layers:
                    out = self.pooling(out)
            layer_num += 1
        if hasattr(self, 'residual_conv'):
            out = out + self.residual_conv(x)
        return torch.transpose(out, 1, 2)

class ResidualTemporalDeConvBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size=[5,3], stride=[2,2], n_groups=8, causal=True, residual=False, pooling_layers=[]):
        super().__init__()
        self.pooling_layers = pooling_layers
        self.blocks = nn.ModuleList()
        for i in range(len(kernel_size)):
            block = DeConv1dBlock(
                inp_channels if i == 0 else out_channels, 
                out_channels, 
                kernel_size[::-1][i], 
                stride[::-1][i], 
                n_groups=n_groups, 
                causal=causal
            )
            self.blocks.append(block)
        if residual:
            if out_channels == inp_channels and stride[0] == 1:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.ConvTranspose1d(inp_channels, out_channels, kernel_size=sum(stride), stride=sum(stride))
        if pooling_layers:
            self.pooling = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

    def forward(self, input_dict):
        x = input_dict
        x = torch.transpose(x, 1, 2)
        out = x
        layer_num = len(self.blocks)-1
        for block in self.blocks:
            if hasattr(self, 'pooling'):
                if layer_num in self.pooling_layers:
                    out = self.pooling(out)
            layer_num -= 1
            out = block(out)
        if hasattr(self, 'residual_conv'):
            out = out + self.residual_conv(x)
        return torch.transpose(out, 1, 2)
    