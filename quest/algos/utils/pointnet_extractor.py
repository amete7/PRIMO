import torch
import torch.nn as nn

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules




class PointNetEncoder(nn.Module):
    """
    Encoder for Pointcloud

    Stolen from DP3 codebase
    """

    def __init__(self,
                 in_shape: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 block_channel=(64, 128, 256, 512),
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()

        in_channels = in_shape[1]
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    



# class DP3Encoder(nn.Module):
#     def __init__(self, 
#                  out_channel=256,
#                  pointcloud_encoder_cfg=None,
#                  use_pc_color=False,
#                  pointnet_type='pointnet',
#                  ):
#         super().__init__()
#         self.point_cloud_key = 'point_cloud'
#         self.n_output_channels = out_channel
        
#         self.use_pc_color = use_pc_color
#         self.pointnet_type = pointnet_type
#         if pointnet_type == "pointnet":
#             if use_pc_color:
#                 pointcloud_encoder_cfg.in_channels = 6
#                 self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
#             else:
#                 pointcloud_encoder_cfg.in_channels = 3
#                 self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
#         else:
#             raise NotImplementedError(f"pointnet_type: {pointnet_type}")



#     def forward(self, observations: Dict) -> torch.Tensor:
#         points = observations[self.point_cloud_key]
#         assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        
#         # points = torch.transpose(points, 1, 2)   # B * 3 * N
#         # points: B * 3 * (N + sum(Ni))
#         pn_feat = self.extractor(points)    # B * out_channel
            
#         return pn_feat


#     def output_shape(self):
#         return self.n_output_channels