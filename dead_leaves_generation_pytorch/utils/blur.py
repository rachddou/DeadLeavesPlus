import torch
import torch.nn.functional as F
import math


def gaussian_blur_torch(img, kernel_size=11, sigma=1.0):
    """
    Apply Gaussian blur to a tensor image.
    
    Args:
        img: torch.Tensor of shape (H, W, C) or (H, W)
        kernel_size: int, size of the Gaussian kernel
        sigma: float, standard deviation of the Gaussian
    
    Returns:
        Blurred image as torch.Tensor
    """
    device = img.device
    dtype = img.dtype
    is_float = dtype.is_floating_point
    
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create Gaussian kernel
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    
    # Convert to float if needed
    if not is_float:
        img = img.float()
    
    # Prepare image for convolution
    if img.dim() == 2:
        img_conv = img.unsqueeze(0).unsqueeze(0)
        kernel_conv = kernel.unsqueeze(0).unsqueeze(0)
        result = F.conv2d(img_conv, kernel_conv, padding=kernel_size//2)
        result = result.squeeze()
    else:
        # (H, W, C) -> (C, H, W)
        img_conv = img.permute(2, 0, 1).unsqueeze(0)
        kernel_conv = kernel.unsqueeze(0).unsqueeze(0).repeat(img.shape[2], 1, 1, 1)
        result = F.conv2d(img_conv, kernel_conv, padding=kernel_size//2, groups=img.shape[2])
        result = result.squeeze(0).permute(1, 2, 0)
    
    # Convert back to original dtype
    if not is_float:
        result = result.to(dtype)
    
    return result


def lens_blur_torch(img, radius=5, components=2, exposure_gamma=2):
    """
    Simplified lens blur approximation using PyTorch.
    
    Args:
        img: torch.Tensor of shape (H, W, C) or (H, W)
        radius: blur radius
        components: number of blur components
        exposure_gamma: exposure correction
    
    Returns:
        Blurred image as torch.Tensor
    """
    device = img.device
    dtype = img.dtype
    is_float = dtype.is_floating_point
    
    # Convert to float [0, 1]
    if not is_float:
        img_float = img.float() / 255.0
    else:
        img_float = img.clone()
    
    # Apply exposure correction
    img_float = torch.pow(img_float.clamp(0, 1), exposure_gamma)
    
    # Apply multiple Gaussian blurs with different sigmas
    result = torch.zeros_like(img_float)
    for i in range(components):
        sigma = radius * (i + 1) / components
        kernel_size = 2 * int(3 * sigma) + 1
        blurred = gaussian_blur_torch(img_float, kernel_size=kernel_size, sigma=sigma)
        result += blurred
    
    result = result / components
    
    # Inverse exposure correction
    result = torch.pow(result.clamp(0, 1), 1.0 / exposure_gamma)
    
    # Convert back to original dtype and range
    if not is_float:
        result = (result * 255).clamp(0, 255).to(dtype)
    
    return result
