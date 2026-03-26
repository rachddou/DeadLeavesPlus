"""GPU smooth geometric warp via torch grid_sample."""
import torch
import torch.nn.functional as F


def generate_perturbation(x: torch.Tensor) -> torch.Tensor:
    """Apply a smooth random displacement field to a 2-D float map.

    Args:
        x: (H, W) float32 tensor.

    Returns:
        (H, W) warped float32 tensor.
    """
    h, w = x.shape
    device = x.device
    s = torch.randint(10, 20, ()).item()
    t = torch.randint(5, s // 2 + 1, ()).item()

    u = torch.randn(1, 1, h, w, device=device)
    v = torch.randn(1, 1, h, w, device=device)

    # Build Gaussian kernel for smoothing
    ks = 61
    kx = torch.arange(ks, dtype=torch.float32, device=device) - ks // 2
    k1d = torch.exp(-0.5 * (kx / float(s)) ** 2)
    k1d = k1d / k1d.sum()
    k2d = k1d.outer(k1d).unsqueeze(0).unsqueeze(0)  # (1,1,ks,ks)
    pad = ks // 2
    u = F.conv2d(F.pad(u, [pad] * 4, mode='reflect'), k2d)
    v = F.conv2d(F.pad(v, [pad] * 4, mode='reflect'), k2d)
    u = (u - u.mean()) / (u.std() + 1e-8) * t
    v = (v - v.mean()) / (v.std() + 1e-8) * t

    # Sampling grid in [-1, 1]
    base_y, base_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing='ij')
    u_n = u[0, 0] / (w / 2.0)
    v_n = v[0, 0] / (h / 2.0)
    grid = torch.stack([base_x + u_n, base_y + v_n], dim=-1).unsqueeze(0)  # (1,H,W,2)

    img = x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    out = F.grid_sample(img, grid, mode='bilinear', padding_mode='border',
                        align_corners=True)[0, 0]
    return out + x.min()
