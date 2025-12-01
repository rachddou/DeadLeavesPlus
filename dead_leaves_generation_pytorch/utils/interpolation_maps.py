import torch
import numpy as np
from dead_leaves_generation_pytorch.utils.utils import rotate_CV, normalize, sigmoid, thresholding
from dead_leaves_generation_pytorch.utils.geometric_perturbation import generate_perturbation
from dead_leaves_generation_pytorch.utils.colored_noise import sample_color_noise


def sample_period(T_min, T_max, n_period, device=None):
    """Sample periods using power distribution."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    periods = torch.floor(T_min + (T_max - T_min) * torch.rand(n_period, device=device) ** (1/2.5))
    return periods


def variable_oscillations(width, T_min, T_max, n_freq, device=None):
    """
    Create pseudo-periodic pattern with variable frequencies in 1D using PyTorch.
    
    Args:
        width: width of final image
        T_min: minimal period
        T_max: maximal period
        n_freq: length of frequency array
        device: torch device
    
    Returns:
        torch.Tensor: 1D oscillation pattern
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    freq_cycles = sample_period(T_min, T_max, n_freq, device=device).long()
    T_full_cycle = freq_cycles.sum().item()
    N_cycles = width // T_full_cycle
    res = torch.zeros(width, dtype=torch.float32, device=device)
    
    start = 0
    for n in range(N_cycles + 1):
        for i in range(n_freq):
            period = freq_cycles[i].item()
            for j in range(period):
                if start + j >= width:
                    return res
                res[start + j] = torch.sin(torch.tensor(((2 * np.pi) / period) * j, device=device))
            start += period
    return res


def sample_grid(width=100, period=[100], angle=45, device=None):
    """
    Create grid pattern with given orientation using PyTorch.
    
    Args:
        width: width of final image
        period: period of grid
        angle: orientation angle
        device: torch device
    
    Returns:
        torch.Tensor: grid pattern
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    thickness = np.random.randint(1, 3)
    x = torch.ones(2*width, device=device)
    
    for i in range(1, 1 + (2*width) // (period[0] + thickness)):
        x[i*(period[0]+thickness)-thickness:i*(period[0]+thickness)] = 0
    
    grid = x.unsqueeze(0).repeat(2*width, 1)
    
    if len(period) == 2:
        y = torch.ones(2*width, device=device)
        for i in range(1, 1 + (2*width) // (period[1] + thickness)):
            y[i*(period[1]+thickness)-thickness:i*(period[1]+thickness)] = 0
        grid_y = y.unsqueeze(0).repeat(2*width, 1).T
        grid = grid * grid_y
    
    grid = normalize(grid[width//2:-width//2, width//2:-width//2])
    grid = rotate_CV(grid, angle, device=device)
    return grid


def sample_sinusoid(width=100, angle=45, angle1=45, angle2=45, variable_freq=False, device=None):
    """
    Create pseudo-periodic grey-level pattern based on sinusoidal functions using PyTorch.
    
    Args:
        width: width of final image
        angle: angle for dimension 1
        angle1: angle for dimension 2
        angle2: rotation applied to whole sinusoidal field
        variable_freq: create sequence of random length periods
        device: torch device
    
    Returns:
        torch.Tensor: sinusoidal pattern
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    T_min = 5
    T_max = 50
    single_dim = np.random.random() > 0.5
    
    if variable_freq:
        sinusoid = variable_oscillations(2*width, T_min, T_max, 20, device=device)
    else:
        period = sample_period(T_min, T_max, 1, device=device)
        sinusoid = torch.sin(((2*np.pi) / period[0]) * torch.arange(0, 2*width, device=device))
    
    sinusoid = rotate_CV(sinusoid.unsqueeze(0).repeat(2*width, 1), angle, device=device)
    
    if not single_dim:
        if variable_freq:
            sinusoid_y = variable_oscillations(2*width, T_min, T_max, 20, device=device)
        else:
            period = sample_period(T_min, T_max, 1, device=device)
            sinusoid_y = torch.sin(((2*np.pi) / period[0]) * torch.arange(0, 2*width, device=device))
        sinusoid_y = sinusoid_y.unsqueeze(0).repeat(2*width, 1).T
        sinusoid_y = rotate_CV(sinusoid_y, angle1, device=device)
        sinusoid = sinusoid * sinusoid_y
    
    lamda = torch.rand(1, device=device).item() * 9 + 1
    sinusoid = sigmoid(sinusoid, lamda, device=device)
    zero_sig = sigmoid(torch.zeros(1, device=device), lamda, device=device)
    one_sig = sigmoid(torch.ones(1, device=device), lamda, device=device)
    sinusoid = (sinusoid - zero_sig) / (one_sig - zero_sig + 1e-8)
    
    sinusoid = 0.5 + 0.5 * sinusoid
    sinusoid = normalize(rotate_CV(sinusoid, angle2, device=device)[width//2:-width//2, width//2:-width//2])
    
    return sinusoid


def sample_interpolation_map(mixing_types=["sin"], width=1000, thresh_val=10, warp=True, device=None):
    """
    Sample interpolation map for texture mixing using PyTorch.
    
    Args:
        mixing_types: list of mixing types ("sin", "grid", "noise")
        width: size of map
        thresh_val: threshold value for noise
        warp: apply warping
        device: torch device
    
    Returns:
        torch.Tensor: interpolation map
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if "sin" in mixing_types:
        angle = np.random.uniform(-45, 45)
        angle1 = angle + np.random.choice([-1, 1]) * np.random.uniform(15, 45)
        angle2 = np.random.uniform(-22.5, 22.5)
        
        sin = sample_sinusoid(width=width, angle=angle, angle1=angle1, angle2=angle2, 
                             variable_freq=(np.random.random() > 0.5), device=device)
        if warp:
            sin = torch.clamp(generate_perturbation(sin, device=device), 0, 1)
    else:
        sin = torch.ones((width, width), device=device)
    
    if "grid" in mixing_types:
        angle_grid = np.random.uniform(-45, 45)
        two_dim = np.random.random() > 0.3
        if two_dim:
            period_grid = [np.random.randint(20, 100), np.random.randint(20, 100)]
        else:
            period_grid = [np.random.randint(20, 100)]
        
        grid = sample_grid(width=width, period=period_grid, angle=angle_grid, device=device)
        if warp:
            grid = torch.clamp(generate_perturbation(grid, device=device), 0, 1)
    else:
        grid = torch.ones((width, width), device=device)
    
    if "noise" in mixing_types:
        pattern = torch.randint(0, 256, (width, width, 3), dtype=torch.uint8, device=device)
        slope_mixing = np.random.uniform(1.5, 3)
        pattern = sample_color_noise(pattern, width, slope_mixing, device=device).float().mean(dim=2)
        pattern = thresholding(pattern, 128 - thresh_val, 128 + thresh_val).float() / 255.0
    else:
        pattern = torch.ones((width, width), device=device)
    
    pattern = grid * sin * pattern
    return pattern