import torch
import torch.nn.functional as F


def rgb2lab_torch(rgb, device=None):
    """
    Convert RGB to LAB color space using PyTorch.
    
    Args:
        rgb: torch.Tensor of shape (..., 3) with values in [0, 1]
        device: torch device
    
    Returns:
        torch.Tensor: LAB color space tensor
    """
    if device is None:
        device = rgb.device if isinstance(rgb, torch.Tensor) else torch.device('cpu')
    
    if not isinstance(rgb, torch.Tensor):
        rgb = torch.from_numpy(rgb).to(device)
    
    rgb = rgb.float()
    
    # RGB to XYZ conversion
    mask = rgb > 0.04045
    rgb_linear = torch.where(mask, torch.pow((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)
    
    # Matrix multiplication for XYZ
    matrix = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], device=device, dtype=torch.float32)
    
    # Reshape for matrix multiplication
    original_shape = rgb.shape
    rgb_reshaped = rgb.reshape(-1, 3)
    xyz = torch.matmul(rgb_reshaped, matrix.T)
    xyz = xyz.reshape(original_shape)
    
    # Normalize by D65 illuminant
    xyz = xyz / torch.tensor([0.95047, 1.0, 1.08883], device=device)
    
    # XYZ to LAB
    mask = xyz > 0.008856
    xyz_f = torch.where(mask, torch.pow(xyz, 1/3), (7.787 * xyz) + (16/116))
    
    L = 116 * xyz_f[..., 1] - 16
    a = 500 * (xyz_f[..., 0] - xyz_f[..., 1])
    b = 200 * (xyz_f[..., 1] - xyz_f[..., 2])
    
    lab = torch.stack([L, a, b], dim=-1)
    return lab


def lab2rgb_torch(lab, device=None):
    """
    Convert LAB to RGB color space using PyTorch.
    
    Args:
        lab: torch.Tensor of shape (..., 3) in LAB color space
        device: torch device
    
    Returns:
        torch.Tensor: RGB color space tensor with values in [0, 1]
    """
    if device is None:
        device = lab.device if isinstance(lab, torch.Tensor) else torch.device('cpu')
    
    if not isinstance(lab, torch.Tensor):
        lab = torch.from_numpy(lab).to(device)
    
    lab = lab.float()
    
    # LAB to XYZ
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    # Inverse transformation
    xyz_f = torch.stack([fx, fy, fz], dim=-1)
    mask = xyz_f > 0.206897  # (6/29)^3
    xyz = torch.where(mask, torch.pow(xyz_f, 3), (xyz_f - 16/116) / 7.787)
    
    # Denormalize by D65
    xyz = xyz * torch.tensor([0.95047, 1.0, 1.08883], device=device)
    
    # XYZ to RGB
    matrix = torch.tensor([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ], device=device, dtype=torch.float32)
    
    original_shape = xyz.shape
    xyz_reshaped = xyz.reshape(-1, 3)
    rgb = torch.matmul(xyz_reshaped, matrix.T)
    rgb = rgb.reshape(original_shape)
    
    # Linear RGB to sRGB
    mask = rgb > 0.0031308
    rgb = torch.where(mask, 1.055 * torch.pow(rgb, 1/2.4) - 0.055, 12.92 * rgb)
    rgb = torch.clamp(rgb, 0, 1)
    
    return rgb
