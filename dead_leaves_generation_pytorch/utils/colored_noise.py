"""GPU-accelerated 1/f^slope coloured noise generation via PyTorch FFT."""
import torch
import torch.fft


def _sample_magnitude_batch(resolution: int, slopes: torch.Tensor) -> torch.Tensor:
    """Batch 1/f^slope magnitude spectra.

    Args:
        resolution: Texture spatial resolution.
        slopes: (B,) float32 slope values.

    Returns:
        (B, 3, resolution, resolution) float32 magnitude maps.
    """
    B = slopes.shape[0]
    device = slopes.device
    idx = torch.arange(resolution, dtype=torch.float32, device=device)
    fx, fy = torch.meshgrid(idx, idx, indexing='xy')          # (H, W)
    fx = (fx - resolution / 2.0) / resolution
    fy = (fy - resolution / 2.0) / resolution

    s = slopes.view(B, 1, 1)                                   # (B,1,1)
    fr = 1e-16 + fx.abs().unsqueeze(0) ** s + fy.abs().unsqueeze(0) ** s  # (B,H,W)
    magnitude = 1.0 / torch.fft.fftshift(fr, dim=(-2, -1))
    magnitude[:, 0, 0] = 0.0
    return magnitude.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()  # (B,3,H,W)


def sample_color_noise_batch(
    patches: list,
    width: int,
    slopes: list,
    device: torch.device,
) -> torch.Tensor:
    """Batch coloured noise textures on GPU.

    Colour palette sampled from source patches; frequency spectrum shaped
    to match the given 1/f^slope.

    Args:
        patches: List of B source patches. Each patch is either a
            (h, w, 3) numpy array or a (h, w, 3) uint8 GPU tensor.
        width: Output texture size (square).
        slopes: List of B float slope values.
        device: Target device.

    Returns:
        (B, 3, width, width) uint8 tensor on device.
    """
    B = len(patches)
    p = 50
    images = torch.zeros(B, 3, width, width, dtype=torch.float32, device=device)

    for b, patch in enumerate(patches):
        h, w = patch.shape[:2]
        xx = torch.randint(0, max(1, h - p), ()).item()
        yy = torch.randint(0, max(1, w - p), ()).item()

        if isinstance(patch, torch.Tensor):
            sub = patch[xx:xx + p, yy:yy + p].reshape(p * p, 3).float().to(device)
            idx = torch.randint(0, p * p, (width * width,), device=device)
            images[b] = sub[idx].T.reshape(3, width, width)
        else:
            import numpy as np
            sub = patch[xx:xx + p, yy:yy + p].reshape(p * p, 3).astype('float32')
            idx = torch.randint(0, p * p, (width * width,)).numpy().astype('int64')
            images[b] = torch.from_numpy(sub[idx]).to(device).T.reshape(3, width, width)

    slopes_t = torch.tensor(slopes, dtype=torch.float32, device=device)
    magnitude = _sample_magnitude_batch(width, slopes_t)       # (B,3,W,W)

    means = images.mean(dim=(-2, -1), keepdim=True)
    stds = images.std(dim=(-2, -1), keepdim=True).clamp(min=1e-8)

    fft_img = torch.fft.fft2(images - means)
    phase = torch.angle(fft_img)
    fft_imposed = magnitude * torch.exp(1j * phase)
    output = torch.fft.ifft2(fft_imposed).real                 # (B,3,W,W)

    out_means = output.mean(dim=(-2, -1), keepdim=True)
    out_stds = output.std(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    output = (output - out_means) / out_stds * stds + means

    return output.clamp(0, 255).to(torch.uint8)
