import torch
import numpy as np
from omegaconf import OmegaConf 
from dead_leaves_generation_pytorch.utils.color_conversion import rgb2lab_torch, lab2rgb_torch
from dead_leaves_generation_pytorch.utils.interpolation_maps import sample_grid, sample_sinusoid, sample_interpolation_map
from dead_leaves_generation_pytorch.utils.geometric_perturbation import generate_perturbation
from dead_leaves_generation_pytorch.utils.colored_noise import sample_color_noise


def _sample_slope_from_ranges(slope_range):
    """
    Échantillonne uniformément une valeur de slope à partir d'une union d'intervalles disjoints.
    
    Args:
        slope_range: soit [a,b] pour un seul intervalle, soit [[a1,b1], [a2,b2], ...] pour plusieurs intervalles
        
    Returns:
        float: sampled slope value
    """
    if isinstance(OmegaConf.to_object(slope_range[0]), (list, tuple)):
        intervals = slope_range
        lengths = [interval[1] - interval[0] for interval in intervals]
        total_length = sum(lengths)
        
        interval_probs = [length / total_length for length in lengths]
        chosen_interval = np.random.choice(len(intervals), p=interval_probs)
        
        interval = intervals[chosen_interval]
        slope = np.random.uniform(interval[0], interval[1])
        return slope
    else:
        slope = np.random.uniform(slope_range[0], slope_range[1])
        return slope


def bilevelTextureMixer(color_source_1=None, color_source_2=None, single_color1=True, single_color2=True,
                       mixing_types=["sin"], width=1000, thresh_val=10, warp=True, slope_range=[0.5, 2.5], device=None):
    """
    Mix two color/texture maps with sinusoidal, grid, or noise pattern using PyTorch.
    
    Args:
        color_source_1: source color image 1 (torch.Tensor or numpy array)
        color_source_2: source color image 2
        single_color1: whether texture 1 is single color or colored noise
        single_color2: whether texture 2 is single color or colored noise
        mixing_types: interpolation methods
        width: size of final texture
        thresh_val: thresholding parameter for noise mixing
        warp: apply warping for sinusoidal interpolation
        slope_range: range for slope parameter
        device: torch device
    
    Returns:
        torch.Tensor: mixed texture image
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Default random sources if not provided
    if color_source_1 is None:
        color_source_1 = torch.randint(0, 256, (100, 100, 3), dtype=torch.uint8, device=device)
    elif not isinstance(color_source_1, torch.Tensor):
        color_source_1 = torch.from_numpy(color_source_1).to(device)
    else:
        color_source_1 = color_source_1.to(device)
    
    if color_source_2 is None:
        color_source_2 = torch.randint(0, 256, (100, 100, 3), dtype=torch.uint8, device=device)
    elif not isinstance(color_source_2, torch.Tensor):
        color_source_2 = torch.from_numpy(color_source_2).to(device)
    else:
        color_source_2 = color_source_2.to(device)
    
    if single_color1:
        idx_h = torch.randint(0, color_source_1.shape[0], (1,), device=device).item()
        idx_w = torch.randint(0, color_source_1.shape[1], (1,), device=device).item()
        texture_map_1 = color_source_1[idx_h, idx_w, :].view(1, 1, 3)
    else:
        slope1 = _sample_slope_from_ranges(slope_range)
        texture_map_1 = sample_color_noise(color_source_1, width, slope1, device=device)
    
    if single_color2:
        idx_h = torch.randint(0, color_source_2.shape[0], (1,), device=device).item()
        idx_w = torch.randint(0, color_source_2.shape[1], (1,), device=device).item()
        texture_map_2 = color_source_2[idx_h, idx_w, :].view(1, 1, 3)
    else:
        slope2 = _sample_slope_from_ranges(slope_range)
        texture_map_2 = sample_color_noise(color_source_2, width, slope2, device=device)
    
    interpolation_map = sample_interpolation_map(mixing_types=mixing_types, width=width, 
                                                 thresh_val=thresh_val, warp=warp, device=device)
    
    # Convert to LAB color space using PyTorch
    texture_map_1 = texture_map_1.float() / 255.0
    texture_map_2 = texture_map_2.float() / 255.0
    
    texture_transform_1 = rgb2lab_torch(texture_map_1, device=device)
    texture_transform_2 = rgb2lab_torch(texture_map_2, device=device)
    
    r = interpolation_map * (texture_transform_2[..., 0] - texture_transform_1[..., 0]) + texture_transform_1[..., 0]
    g = interpolation_map * (texture_transform_2[..., 1] - texture_transform_1[..., 1]) + texture_transform_1[..., 1]
    b = interpolation_map * (texture_transform_2[..., 2] - texture_transform_1[..., 2]) + texture_transform_1[..., 2]
    
    res_image = torch.stack([r, g, b], dim=-1)
    
    # Convert back to RGB using PyTorch
    res_image = lab2rgb_torch(res_image, device=device)
    res_image = (res_image * 255.0).clamp(0, 255).byte()
    
    return res_image


def pattern_patch_two_colors(color_1, color_2, width=100, period=[100], thickness=3, angle=45, 
                            warp=False, type="sin", device=None):
    """
    Mix two color maps with sinusoidal pattern using PyTorch.
    
    Args:
        color_1: either an image or single RGB color
        color_2: either an image or single RGB color
        width: image width
        period: period of oscillations
        thickness: thickness of grid
        angle: rotation angle
        warp: apply atmospheric perturbation
        type: "sin" or "grid"
        device: torch device
    
    Returns:
        torch.Tensor: pattern image
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not isinstance(color_1, torch.Tensor):
        color_1 = torch.tensor(color_1, device=device, dtype=torch.float32)
    else:
        color_1 = color_1.float().to(device)
    
    if not isinstance(color_2, torch.Tensor):
        color_2 = torch.tensor(color_2, device=device, dtype=torch.float32)
    else:
        color_2 = color_2.float().to(device)
    
    color_1 = color_1 / 255.0
    color_2 = color_2 / 255.0
    
    # Convert to LAB using PyTorch
    texture_transform_1 = rgb2lab_torch(color_1, device=device)
    texture_transform_2 = rgb2lab_torch(color_2, device=device)
    
    if type == "sin":
        pattern = sample_sinusoid(width=width, angle=angle, variable_freq=(np.random.random() > 0.5), device=device)
    elif type == "grid":
        pattern = sample_grid(width=width, period=period, angle=angle, device=device)
    
    if warp:
        pattern = torch.clamp(generate_perturbation(pattern, device=device), 0, 1)
    
    r = pattern * (texture_transform_2[..., 0] - texture_transform_1[..., 0]) + texture_transform_1[..., 0]
    g = pattern * (texture_transform_2[..., 1] - texture_transform_1[..., 1]) + texture_transform_1[..., 1]
    b = pattern * (texture_transform_2[..., 2] - texture_transform_1[..., 2]) + texture_transform_1[..., 2]
    
    res_image = torch.stack([r, g, b], dim=-1)
    
    # Convert back to RGB using PyTorch
    res_image = lab2rgb_torch(res_image, device=device)
    res_image = (res_image * 255.0).clamp(0, 255).byte()
    
    return res_image