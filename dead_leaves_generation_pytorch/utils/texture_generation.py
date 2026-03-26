"""GPU-accelerated bilevel texture mixing in CIE-LAB space."""
import numpy as np
import torch

from dead_leaves_generation_pytorch.utils.colored_noise import sample_color_noise_batch
from dead_leaves_generation_pytorch.utils.interpolation_maps import sample_interpolation_map
from dead_leaves_generation_pytorch.utils.utils import rgb_to_lab_torch, lab_to_rgb_torch


def _sample_slope_from_ranges(slope_range) -> float:
    if not isinstance(slope_range[0], (int, float)):
        intervals = slope_range
        lengths = [iv[1] - iv[0] for iv in intervals]
        total = sum(lengths)
        probs = [l / total for l in lengths]
        idx = np.random.choice(len(intervals), p=probs)
        return float(np.random.uniform(intervals[idx][0], intervals[idx][1]))
    return float(np.random.uniform(slope_range[0], slope_range[1]))


def bilevelTextureMixer_batch(
    color_sources_1: list,
    color_sources_2: list,
    single_color1: bool = True,
    single_color2: bool = True,
    mixing_types: list = None,
    width: int = 1000,
    thresh_val: int = 10,
    warp: bool = True,
    slope_range=None,
    device: torch.device = None,
) -> torch.Tensor:
    """Mix two colour/texture sources for a batch of textures on GPU.

    Args:
        color_sources_1: List of B source patches (numpy arrays h×w×3).
        color_sources_2: List of B source patches.
        single_color1: Use a single colour from source 1 (True) or generate
            a coloured-noise map (False).
        single_color2: Same for source 2.
        mixing_types: Subset of ["sin", "grid", "noise"]. Defaults to ["sin"].
        width: Output texture size.
        thresh_val: Threshold half-width for noise binarisation.
        warp: Apply geometric warp to the interpolation maps.
        slope_range: 1/f slope range; [a, b] or [[a1,b1], ...].
        device: Target device.

    Returns:
        (B, 3, width, width) uint8 tensor on device.
    """
    if mixing_types is None:
        mixing_types = ["sin"]
    if slope_range is None:
        slope_range = [0.5, 2.5]
    if device is None:
        device = torch.device('cpu')
    B = len(color_sources_1)

    def _pick_colors(sources: list, single: bool) -> torch.Tensor:
        """Returns (B, 3, W, W) float32."""
        if single:
            cols = []
            for src in sources:
                r = np.random.randint(0, src.shape[0])
                c = np.random.randint(0, src.shape[1])
                cols.append(torch.from_numpy(src[r, c].astype(np.float32)).to(device))
            return torch.stack(cols).view(B, 3, 1, 1).expand(B, 3, width, width).contiguous()
        else:
            slopes = [_sample_slope_from_ranges(slope_range) for _ in range(B)]
            return sample_color_noise_batch(sources, width, slopes, device).float()

    tm1 = _pick_colors(color_sources_1, single_color1)  # (B, 3, W, W) float
    tm2 = _pick_colors(color_sources_2, single_color2)

    # One interpolation map per batch item (each is different)
    interp = torch.stack([
        sample_interpolation_map(mixing_types=mixing_types, width=width,
                                 thresh_val=thresh_val, warp=warp, device=device)
        for _ in range(B)
    ]).unsqueeze(1)  # (B, 1, W, W)

    # CIE-LAB mixing  →  (B, W, W, 3)
    tm1_hwc = (tm1 / 255.0).permute(0, 2, 3, 1)
    tm2_hwc = (tm2 / 255.0).permute(0, 2, 3, 1)
    t1_lab = rgb_to_lab_torch(tm1_hwc)
    t2_lab = rgb_to_lab_torch(tm2_hwc)
    interp_hwc = interp.permute(0, 2, 3, 1)                # (B, W, W, 1)
    mixed_lab = t1_lab + interp_hwc * (t2_lab - t1_lab)
    mixed_rgb = lab_to_rgb_torch(mixed_lab)                 # (B, W, W, 3) in [0,1]

    return (mixed_rgb * 255.0).clamp(0, 255).to(torch.uint8).permute(0, 3, 1, 2)  # (B, 3, W, W)
