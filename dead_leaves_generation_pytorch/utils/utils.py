"""GPU utility functions: rotation, normalisation, colour space, gradients."""
import numpy as np
import torch
import torch.nn.functional as F


def rotate_torch(image: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Centre-rotate a 2-D float tensor by angle_deg degrees using affine_grid.
    Handles (H,W), (C,H,W) and (B,C,H,W) inputs; returns the same shape."""
    squeeze_dims = []
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
        squeeze_dims = [0, 1]
    elif image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze_dims = [0]
    angle_rad = angle_deg * np.pi / 180.0
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot = torch.tensor(
        [[[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]]],
        dtype=torch.float32, device=image.device
    ).expand(image.shape[0], -1, -1)
    grid = F.affine_grid(rot, image.shape, align_corners=False)
    out = F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    for d in reversed(squeeze_dims):
        out = out.squeeze(d)
    return out


def normalize_torch(x: torch.Tensor) -> torch.Tensor:
    """Normalise tensor to [0, 1]."""
    xmin, xmax = x.min(), x.max()
    return (x - xmin) / (xmax - xmin + 1e-8)


def sigmoid_torch(x: torch.Tensor, lamda: float) -> torch.Tensor:
    """Sigmoid with tunable steepness; hard-clips when lamda >= 20."""
    sig = torch.sigmoid(torch.as_tensor(lamda, dtype=x.dtype, device=x.device) * x)
    if lamda >= 20:
        sig = (sig > 0.5).to(x.dtype)
    return sig


def rgb_to_lab_torch(rgb: torch.Tensor) -> torch.Tensor:
    """Convert an (..., 3) float32 RGB tensor in [0, 1] to CIE-LAB via kornia.
    Returns the same shape in LAB space."""
    import kornia.color as KC
    shape = rgb.shape
    x = rgb.reshape(-1, 1, 1, 3).permute(0, 3, 1, 2).contiguous()  # (N,3,1,1)
    lab = KC.rgb_to_lab(x)  # (N,3,1,1)
    return lab.permute(0, 2, 3, 1).reshape(shape)


def lab_to_rgb_torch(lab: torch.Tensor) -> torch.Tensor:
    """Convert an (..., 3) CIE-LAB tensor to float32 RGB in [0, 1] via kornia."""
    import kornia.color as KC
    shape = lab.shape
    x = lab.reshape(-1, 1, 1, 3).permute(0, 3, 1, 2).contiguous()  # (N,3,1,1)
    rgb = KC.lab_to_rgb(x)  # (N,3,1,1)
    return rgb.permute(0, 2, 3, 1).reshape(shape).clamp(0.0, 1.0)


def linear_color_gradient_torch(
    color_1: torch.Tensor,
    color_2: torch.Tensor,
    width: int,
    angle: float,
    k: float = 0.5,
    device: torch.device = None,
) -> torch.Tensor:
    """Smooth LAB colour gradient between two RGB colours.

    Args:
        color_1: (3,) uint8 tensor.
        color_2: (3,) uint8 tensor.
        width: Output image size (square).
        angle: Gradient angle in degrees.
        k: Logistic smoothness parameter.
        device: Target device.

    Returns:
        (width, width, 3) uint8 tensor.
    """
    if device is None:
        device = color_1.device
    lin = torch.linspace(0, 1, 2 * width, device=device)
    grad = lin.unsqueeze(0).expand(2 * width, -1)           # (2W, 2W)
    grad = normalize_torch(
        rotate_torch(grad, angle)[width // 2:-width // 2, width // 2:-width // 2])

    def _norm_logistic(x: torch.Tensor, kk: float) -> torch.Tensor:
        def _l(t):
            return 1.0 / (1.0 + torch.exp(-kk * (12.1 * t - 6.0)))
        lo = _l(torch.zeros(1, device=device))
        hi = _l(torch.ones(1, device=device) * 1.1)
        return (_l(x) - lo) / (hi - lo + 1e-8)

    grad = _norm_logistic(grad, k)                          # (W, W)

    c1 = color_1.float() / 255.0
    c2 = color_2.float() / 255.0
    c1_lab = rgb_to_lab_torch(c1.view(1, 1, 3))             # (1,1,3)
    c2_lab = rgb_to_lab_torch(c2.view(1, 1, 3))

    g = grad.unsqueeze(-1)                                  # (W,W,1)
    mixed_lab = c1_lab + g * (c2_lab - c1_lab)             # (W,W,3)
    mixed_rgb = lab_to_rgb_torch(mixed_lab)                 # (W,W,3) in [0,1]
    return (mixed_rgb * 255.0).clamp(0, 255).to(torch.uint8)
