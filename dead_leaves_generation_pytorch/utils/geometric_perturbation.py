import torch
import torch.nn.functional as F
import numpy as np
from dead_leaves_generation_pytorch.utils.blur import gaussian_blur_torch


def generate_triangular_pattern(shape, T, device=None):
    """Generate triangular pattern using PyTorch."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tmp = shape / T
    n_period = int(tmp)
    pattern = torch.zeros(shape, dtype=torch.float32, device=device)
    
    for i in range(n_period):
        for j in range(T//2):
            if i*T + j >= shape:
                return pattern
            pattern[i*T + j] = 4*j/T - 1.0
        for j in range(T//2):
            if i*T + T//2 + j >= shape:
                return pattern
            pattern[i*T + T//2 + j] = 4*(T//2 - j)/T - 1.0
    return pattern


def generate_vector_field(shape, device=None):
    """
    Generate direction maps for distortion parameter using PyTorch.
    
    Args:
        shape: side length of square
        device: torch device
    
    Returns:
        tuple: (u, v) direction maps
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    u = torch.zeros((shape, shape), device=device)
    v = torch.ones((shape, shape), device=device)
    
    type = "athmos"
    
    if type == "athmos":
        u = torch.randn(shape, shape, device=device)
        v = torch.randn(shape, shape, device=device)
        
        s = np.random.randint(10, 20)
        t = np.random.randint(5, s//2 + 1)
        
        u = gaussian_blur_torch(u, kernel_size=61, sigma=s)
        v = gaussian_blur_torch(v, kernel_size=61, sigma=s)
        
        u = (u - u.mean()) * (t / (u.std() + 1e-8))
        v = (v - v.mean()) * (t / (v.std() + 1e-8))
    elif type == "sin":
        T = np.random.randint(50, 1000)
        intensity = np.random.uniform(0.8, 0.1*T*1.2)
        sinusoid = intensity * torch.sin(2*np.pi*torch.arange(0, shape, device=device)/T).view(shape, 1)
        u = sinusoid.repeat(1, shape)
    elif type == "triangle":
        T = np.random.randint(50, 1000)
        intensity = np.random.uniform(0.1*T, 0.5*T)
        pattern = generate_triangular_pattern(shape, T, device=device).view(shape, 1)
        pattern = intensity * pattern
        u = pattern.repeat(1, shape)
    
    return u, v


def bilinear_interpolate(im, xx, yy):
    """
    Bilinear interpolation using PyTorch.
    
    Args:
        im: image tensor (H, W)
        xx: x interpolation parameters
        yy: y interpolation parameters
    
    Returns:
        torch.Tensor: interpolated image
    """
    device = im.device
    
    x0 = torch.floor(xx).long()
    x1 = x0 + 1
    y0 = torch.floor(yy).long()
    y1 = y0 + 1
    
    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)
    
    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]
    
    wa = (x1.float() - xx) * (y1.float() - yy)
    wb = (x1.float() - xx) * (yy - y0.float())
    wc = (xx - x0.float()) * (y1.float() - yy)
    wd = (xx - x0.float()) * (yy - y0.float())
    
    return wa*Ia + wb*Ib + wc*Ic + wd*Id


def generate_perturbation(x, device=None):
    """
    Generate distortion perturbation using PyTorch.
    
    Args:
        x: original grey level image (torch.Tensor)
        device: torch device
    
    Returns:
        torch.Tensor: distorted image
    """
    if device is None:
        device = x.device if isinstance(x, torch.Tensor) else torch.device('cpu')
    
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).to(device)
    
    shape = x.shape[0]
    u, v = generate_vector_field(shape, device=device)
    
    xx, yy = torch.meshgrid(torch.arange(shape, device=device), 
                            torch.arange(shape, device=device), indexing='xy')
    
    res = bilinear_interpolate(x, u + xx.float(), v + yy.float()) + x.min()
    
    return res