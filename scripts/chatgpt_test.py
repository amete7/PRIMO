import torch
import numpy as np

def batch_axis_angle_to_rotation_matrix(axes, angles):
    """
    Convert a batch of axis-angle representations to rotation matrices.

    Parameters:
    axes (torch.Tensor): A tensor of shape (N, 3) representing the axes of rotation (must be unit vectors).
    angles (torch.Tensor): A tensor of shape (N,) representing the angles of rotation in radians.

    Returns:
    torch.Tensor: A tensor of shape (N, 3, 3) containing the rotation matrices.
    """
    # Ensure the axes are unit vectors
    axes = axes / torch.norm(axes, dim=1, keepdim=True)
    x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]
    
    # Compute the skew-symmetric matrices K for each axis
    zeros = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zeros, -z, y], dim=1),
        torch.stack([z, zeros, -x], dim=1),
        torch.stack([-y, x, zeros], dim=1)
    ], dim=1)
    
    # Compute the identity matrix for each batch
    I = torch.eye(3, device=axes.device).unsqueeze(0).repeat(axes.size(0), 1, 1)
    
    # Reshape angles for broadcasting
    angles = angles.view(-1, 1, 1)
    
    # Compute the rotation matrices using Rodrigues' rotation formula
    R = I + torch.sin(angles) * K + (1 - torch.cos(angles)) * torch.matmul(K, K)
    
    return R

# Example usage
batch_size = 4
axes = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=torch.float32).cuda()  # Example batch of axes
angles = torch.tensor([np.pi / 4, np.pi / 2, np.pi, np.pi / 3], dtype=torch.float32).cuda()  # Example batch of angles

rotation_matrices = batch_axis_angle_to_rotation_matrix(axes, angles)
print(rotation_matrices.shape)
