import torch
import torch.nn.functional as F
import numpy as np


def perspective_transform_torch(image, src_points, dst_points, device=None):
    """
    Apply perspective transform using PyTorch.
    
    Args:
        image: torch.Tensor of shape (H, W, C) or (H, W)
        src_points: list of 4 (x, y) tuples - source points
        dst_points: list of 4 (x, y) tuples - destination points
        device: torch device
    
    Returns:
        torch.Tensor: transformed image
    """
    if device is None:
        device = image.device if isinstance(image, torch.Tensor) else torch.device('cpu')
    
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).to(device)
    
    # Convert points to tensors
    src = torch.tensor(src_points, dtype=torch.float32, device=device)
    dst = torch.tensor(dst_points, dtype=torch.float32, device=device)
    
    # Solve for homography matrix using DLT (Direct Linear Transform)
    # Build matrix A for the equation Ah = 0
    A = []
    for i in range(4):
        x, y = src[i]
        u, v = dst[i]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    
    A = torch.stack([torch.tensor(row, device=device) for row in A])
    
    # SVD to find homography
    U, S, Vh = torch.linalg.svd(A)
    H = Vh[-1].reshape(3, 3)
    H = H / H[2, 2]  # Normalize
    
    # Convert homography to affine grid format for grid_sample
    # We need to invert H and convert to normalized coordinates
    H_inv = torch.linalg.inv(H)
    
    # Prepare image
    h, w = image.shape[:2]
    if image.dim() == 2:
        img_4d = image.unsqueeze(0).unsqueeze(0).float()
    else:
        img_4d = image.permute(2, 0, 1).unsqueeze(0).float()
    
    # Create grid of output coordinates
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing='ij'
    )
    
    # Convert to homogeneous coordinates (in pixel space)
    ones = torch.ones_like(grid_x)
    grid_x_px = (grid_x + 1) * (w - 1) / 2
    grid_y_px = (grid_y + 1) * (h - 1) / 2
    
    coords = torch.stack([grid_x_px, grid_y_px, ones], dim=-1)
    
    # Apply inverse homography
    coords_t = torch.matmul(coords, H_inv.T)
    coords_t = coords_t[..., :2] / (coords_t[..., 2:3] + 1e-8)
    
    # Convert back to normalized coordinates [-1, 1]
    grid_sample = torch.stack([
        coords_t[..., 0] * 2 / (w - 1) - 1,
        coords_t[..., 1] * 2 / (h - 1) - 1
    ], dim=-1)
    
    # Sample from input image
    output = F.grid_sample(img_4d, grid_sample.unsqueeze(0), 
                          mode='bilinear', padding_mode='zeros', align_corners=True)
    
    # Convert back to original format
    if image.dim() == 2:
        return output.squeeze()
    else:
        return output.squeeze(0).permute(1, 2, 0)


def perspective_shift(image, device=None):
    """
    Apply perspective shift to image using PyTorch.
    
    Args:
        image: torch.Tensor of shape (H, W, C)
        device: torch device
    
    Returns:
        torch.Tensor: perspective-shifted image
    """
    if device is None:
        device = image.device if isinstance(image, torch.Tensor) else torch.device('cpu')
    
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).to(device)
    
    image = image.to(device)
    
    size = image.shape[0]
    alpha = 2
    beta = int(size // 4)
    mode = np.random.randint(0, 3)
    
    src_points = [(0, 0), (0, size), (size//alpha, size - beta), (size//alpha, beta)]
    dst_points = [(0, 0), (0, size), (size, size), (size, 0)]
    
    # Apply perspective transform
    img_transformed = perspective_transform_torch(image, src_points, dst_points, device=device)
    
    # Crop result
    result = img_transformed[beta:size-beta, 0:size//alpha]
    
    if mode == 0:
        result = torch.flip(result, [1])
    elif mode == 1:
        result = torch.rot90(result, k=1, dims=(0, 1))
    
    return result