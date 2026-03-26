"""GPU perspective shift using kornia."""
import torch


def perspective_shift(image: torch.Tensor) -> torch.Tensor:
    """Apply a random perspective shift to an image tensor.

    Args:
        image: (H, W, 3) uint8 tensor.

    Returns:
        Perspective-shifted and cropped tensor.
    """
    import kornia.geometry.transform as KGT
    size = image.shape[0]
    device = image.device
    alpha, beta = 2, int(size // 4)
    mode = torch.randint(0, 3, ()).item()

    src = torch.tensor([
        [0.0, 0.0], [0.0, float(size)],
        [float(size // alpha), float(size - beta)],
        [float(size // alpha), float(beta)],
    ], dtype=torch.float32, device=device).unsqueeze(0)
    dst = torch.tensor([
        [0.0, 0.0], [0.0, float(size)],
        [float(size), float(size)], [float(size), 0.0],
    ], dtype=torch.float32, device=device).unsqueeze(0)

    M = KGT.get_perspective_transform(src, dst)             # (1, 3, 3)
    img_t = image.permute(2, 0, 1).unsqueeze(0).float()    # (1, 3, H, W)
    warped = KGT.warp_perspective(img_t, M, dsize=(size, size),
                                  mode='bicubic', padding_mode='zeros')
    result = warped[0].permute(1, 2, 0).clamp(0, 255).to(torch.uint8)
    result = result[beta:size - beta, 0:size // alpha]
    if mode == 0:
        result = result.flip(1)
    elif mode == 1:
        result = result.rot90(1, [0, 1])
    return result
