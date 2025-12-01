import torch
import numpy as np


def sample_magnitude(resolution, slope, device=None):
    """
    Create magnitude map for frequency domain in 1/f^slope using PyTorch.
    
    Args:
        resolution: size of texture image
        slope: slope of magnitude map in log domain
        device: torch device
    
    Returns:
        torch.Tensor: magnitude map
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    fx, fy = torch.meshgrid(torch.arange(resolution, device=device), 
                            torch.arange(resolution, device=device), indexing='ij')
    
    fx = fx.float() - resolution / 2
    fy = fy.float() - resolution / 2
    
    fr = 1e-16 + torch.abs(fx / resolution) ** slope + torch.abs(fy / resolution) ** slope
    
    magnitude = 1 / torch.fft.fftshift(fr)
    magnitude[0, 0] = 0
    magnitude = magnitude.unsqueeze(-1).repeat(1, 1, 3)
    
    return magnitude


def sample_color_noise(image, width, slope, device=None):
    """
    Create colored noise with given slope from color histogram using PyTorch.
    
    Args:
        image: source image for color histogram (torch.Tensor or numpy array)
        width: size of texture image
        slope: slope of magnitude map in log domain
        device: torch device
    
    Returns:
        torch.Tensor: colored noise image
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).to(device)
    
    h, w = image.shape[0], image.shape[1]
    p = 50
    
    xx = torch.randint(0, max(1, h - p), (1,), device=device).item()
    yy = torch.randint(0, max(1, w - p), (1,), device=device).item()
    image = image[xx:xx+p, yy:yy+p]
    image = image.reshape(p*p, 3)
    
    index = torch.randint(0, p*p, (width*width,), device=device)
    image = image[index]
    image = image.reshape(width, width, 3).float()
    
    magnitude = sample_magnitude(width, slope, device=device)
    
    for c in range(3):
        x = image[..., c]
        
        m = x.mean()
        s = x.std()
        
        fft_img = torch.fft.fft2(x - m)
        fft_phase = torch.angle(fft_img)
        
        fft_imposed = magnitude[..., c] * torch.exp(1j * fft_phase)
        y = torch.real(torch.fft.ifft2(fft_imposed))
        
        y = y - y.mean()
        y = y / (y.std() + 1e-8) * s + m
        
        image[..., c] = y
    
    image = torch.clamp(image, 0, 255).byte()
    return image