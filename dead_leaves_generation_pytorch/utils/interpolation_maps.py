"""GPU sinusoidal and grid interpolation maps for texture mixing."""
import numpy as np
import torch

from dead_leaves_generation_pytorch.utils.utils import rotate_torch, normalize_torch, sigmoid_torch

TMIN = 5
TMAX = 100


def sample_grid(
    width: int = 100,
    period: list = None,
    angle: float = 45.0,
    device: torch.device = None,
) -> torch.Tensor:
    """Binary grid interpolation map.

    Args:
        width: Output size.
        period: List of 1 or 2 period values (pixels).
        angle: Rotation angle in degrees.
        device: Target device.

    Returns:
        (width, width) float32 tensor in [0, 1].
    """
    if period is None:
        period = [100]
    if device is None:
        device = torch.device('cpu')
    thickness = int(np.random.randint(1, 3))
    two_w = 2 * width

    x = torch.ones(two_w, dtype=torch.float32, device=device)
    for i in range(1, 1 + two_w // (period[0] + thickness)):
        x[i * (period[0] + thickness) - thickness: i * (period[0] + thickness)] = 0.0
    grid = x.unsqueeze(0).expand(two_w, -1)  # (2W, 2W)

    if len(period) == 2:
        y = torch.ones(two_w, dtype=torch.float32, device=device)
        for i in range(1, 1 + two_w // (period[1] + thickness)):
            y[i * (period[1] + thickness) - thickness: i * (period[1] + thickness)] = 0.0
        grid = grid * y.unsqueeze(1)

    grid = normalize_torch(grid[width // 2:-width // 2, width // 2:-width // 2])
    return rotate_torch(grid, angle)


def _build_sinusoid_1d(two_w: int, variable_freq: bool, device: torch.device) -> torch.Tensor:
    """Build a 1-D sinusoidal signal of length two_w."""
    if variable_freq:
        periods = np.floor(TMIN + (TMAX - TMIN) * np.random.random(20)).astype(int)
        vec = np.zeros(two_w, dtype=np.float32)
        start, n = 0, 0
        while start < two_w:
            p = max(1, int(periods[n % len(periods)]))
            for j in range(p):
                if start + j >= two_w:
                    break
                vec[start + j] = np.sin(2.0 * np.pi * j / p)
            start += p
            n += 1
        return torch.from_numpy(vec).to(device)
    else:
        period = int(np.floor(TMIN + (TMAX - TMIN) * np.random.random()))
        period = max(1, period)
        t = torch.arange(two_w, dtype=torch.float32, device=device)
        return torch.sin(2.0 * np.pi * t / period)


def sample_sinusoid(
    width: int = 100,
    angle: float = 45.0,
    angle1: float = 45.0,
    angle2: float = 45.0,
    variable_freq: bool = False,
    device: torch.device = None,
) -> torch.Tensor:
    """Pseudo-periodic sinusoidal interpolation map.

    Args:
        width: Output size.
        angle: Rotation of the first sinusoid dimension.
        angle1: Rotation of the optional second dimension.
        angle2: Final global rotation.
        variable_freq: Use variable-frequency oscillations.
        device: Target device.

    Returns:
        (width, width) float32 tensor in [0, 1].
    """
    if device is None:
        device = torch.device('cpu')
    two_w = 2 * width
    single_dim = np.random.random() > 0.5
    thresh = np.random.uniform(0.05, 1.0, 2)
    lamda = np.random.uniform(1.0, 10.0, 2)

    sin1d = _build_sinusoid_1d(two_w, variable_freq, device)
    sin_2d = sin1d.unsqueeze(0).expand(two_w, -1)  # (2W, 2W)
    sin_2d = rotate_torch(sin_2d, angle)
    sin_2d = 0.5 + 0.5 * sin_2d
    sin_2d = sin_2d.clone()
    sin_2d[sin_2d > thresh[0]] = 1.0
    sin_2d = sigmoid_torch(sin_2d, lamda[0])
    lo = sigmoid_torch(torch.zeros(1, device=device), lamda[0])
    hi = sigmoid_torch(torch.ones(1, device=device), lamda[0])
    sin_2d = (sin_2d - lo) / (hi - lo + 1e-8)

    if not single_dim:
        sin1d_y = _build_sinusoid_1d(two_w, variable_freq, device)
        sin_y = sin1d_y.unsqueeze(1).expand(-1, two_w)  # (2W, 2W)
        sin_y = rotate_torch(sin_y, angle1)
        sin_y = 0.5 + 0.5 * sin_y
        sin_y = sin_y.clone()
        sin_y[sin_y > thresh[1]] = 1.0
        sin_y = sigmoid_torch(sin_y, lamda[1])
        lo_y = sigmoid_torch(torch.zeros(1, device=device), lamda[1])
        hi_y = sigmoid_torch(torch.ones(1, device=device), lamda[1])
        sin_y = (sin_y - lo_y) / (hi_y - lo_y + 1e-8)
        sin_2d = sin_2d * sin_y

    cropped = rotate_torch(sin_2d, angle2)[width // 2:-width // 2, width // 2:-width // 2]
    return normalize_torch(cropped)


def sample_interpolation_map(
    mixing_types: list = None,
    width: int = 1000,
    thresh_val: int = 10,
    warp: bool = True,
    device: torch.device = None,
) -> torch.Tensor:
    """Composite interpolation map combining sin, grid, and noise components.

    Args:
        mixing_types: Subset of ["sin", "grid", "noise"]. Defaults to ["sin"].
        width: Output size.
        thresh_val: Threshold half-width for noise binarisation.
        warp: Apply geometric warp to each component.
        device: Target device.

    Returns:
        (width, width) float32 tensor in [0, 1].
    """
    if mixing_types is None:
        mixing_types = ["sin"]
    if device is None:
        device = torch.device('cpu')

    from dead_leaves_generation_pytorch.utils.geometric_perturbation import generate_perturbation

    if "sin" in mixing_types:
        ang = np.random.uniform(-45, 45)
        ang1 = ang + np.random.choice([-1, 1]) * np.random.uniform(15, 45)
        ang2 = np.random.uniform(-22.5, 22.5)
        sin = sample_sinusoid(width=width, angle=ang, angle1=ang1, angle2=ang2,
                              variable_freq=False, device=device)
        if warp:
            sin = generate_perturbation(sin).clamp(0, 1)
    else:
        sin = torch.ones(width, width, dtype=torch.float32, device=device)

    if "grid" in mixing_types:
        ang_g = float(np.random.uniform(-45, 45))
        two_dim = np.random.random() > 0.3
        periods = ([int(np.random.randint(20, 100)), int(np.random.randint(20, 100))]
                   if two_dim else [int(np.random.randint(20, 100))])
        grid = sample_grid(width=width, period=periods, angle=ang_g, device=device)
        if warp:
            grid = generate_perturbation(grid).clamp(0, 1)
    else:
        grid = torch.ones(width, width, dtype=torch.float32, device=device)

    if "noise" in mixing_types:
        from dead_leaves_generation_pytorch.utils.colored_noise import sample_color_noise_batch
        rnd = np.random.randint(0, 255, (width, width, 3), dtype=np.uint8)
        slope_m = float(np.random.uniform(1.5, 3.0))
        noise_batch = sample_color_noise_batch([rnd], width, [slope_m], device)
        noise_map = noise_batch[0].float().mean(dim=0)           # (W, W)
        lo_t = float(128 - thresh_val)
        hi_t = float(128 + thresh_val)
        noise_map = (noise_map.clamp(lo_t, hi_t) - lo_t) / (hi_t - lo_t + 1e-8)
    else:
        noise_map = torch.ones(width, width, dtype=torch.float32, device=device)

    return (grid * sin * noise_map).clamp(0, 1)
