import numpy as np
import torch
import torch.nn.functional as F
import math
import random


def rotate_tensor(img, angle, device=None):
    """
    Rotate a tensor by a given angle using PyTorch.
    
    Args:
        img: torch.Tensor of shape (H, W)
        angle: rotation angle in degrees
        device: torch device
    
    Returns:
        torch.Tensor: rotated image
    """
    if device is None:
        device = img.device if isinstance(img, torch.Tensor) else torch.device('cpu')
    
    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img).to(device)
    
    angle_rad = torch.tensor(angle * math.pi / 180.0, device=device)
    
    # Get image dimensions
    h, w = img.shape
    
    # Create rotation matrix
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    
    # Calculate new image size
    new_h = int(abs(h * cos_a) + abs(w * sin_a)) + 1
    new_w = int(abs(w * cos_a) + abs(h * sin_a)) + 1
    
    # Make new_h and new_w odd
    new_h = new_h if new_h % 2 == 1 else new_h + 1
    new_w = new_w if new_w % 2 == 1 else new_w + 1
    
    # Create affine transformation matrix
    theta = torch.zeros(1, 2, 3, device=device)
    theta[0, 0, 0] = cos_a
    theta[0, 0, 1] = sin_a
    theta[0, 1, 0] = -sin_a
    theta[0, 1, 1] = cos_a
    
    # Prepare image for grid_sample
    img_4d = img.unsqueeze(0).unsqueeze(0).float()
    
    # Create grid for new size
    grid = F.affine_grid(theta, (1, 1, new_h, new_w), align_corners=False)
    
    # Scale grid to original image coordinates
    grid[:, :, :, 0] = grid[:, :, :, 0] * w / new_w
    grid[:, :, :, 1] = grid[:, :, :, 1] * h / new_h
    
    # Apply transformation
    rotated = F.grid_sample(img_4d, grid, mode='nearest', align_corners=False)
    
    return rotated.squeeze()


def make_rectangle_mask(radius, device=None):
    """Generates a binary mask of a rectangle of area = pi*radius^2."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    area = math.pi * (radius ** 2)
    ratio = random.uniform(0.5, 1)
    width = int(math.sqrt(ratio * area))
    length = int(math.sqrt(area / ratio))
    
    mask = torch.zeros((length, length), device=device)
    mask[int(length//2 - width//2):int(length//2 + width//2), :] = 1
    
    angle = random.uniform(0, 180)
    mask = rotate_tensor(mask, angle, device=device)
    mask = mask > 0.5
    
    h, w = mask.shape
    h_odd = 2 * (h // 2) - 1
    w_odd = 2 * (w // 2) - 1
    
    mask = mask[:h_odd, :w_odd]
    return mask.bool()


def sample_points_circle(n, radius, device=None):
    """Generates n points sampled uniformly in a circle of radius radius."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    distance_to_center = torch.randint(0, radius, (n,), device=device).float()
    angle = torch.rand(n, device=device) * 2 * math.pi - math.pi
    
    x = radius + distance_to_center * torch.cos(angle)
    y = radius + distance_to_center * torch.sin(angle)
    return torch.stack([x, y], dim=-1)


def concave_hull(points: torch.Tensor, ratio: float = 0.1, max_k: int = 50):
    """
    GPU Concave Hull approximation (k-NN based).
    Shapely-like interface:
        concave_hull(points, ratio=0.1)

    Parameters
    ----------
    points : (N,2) tensor (float32/float64) on GPU
    ratio  : float in (0,1], smaller = more concave 
    max_k  : cap for neighborhood search

    Returns
    -------
    hull : (M,2) tensor of hull vertices (GPU)
    """
    device = points.device
    N = points.shape[0]

    k = max(3, min(max_k, int(N * ratio)))

    # Pairwise distances
    diff = points[:, None, :] - points[None, :, :]
    dist = torch.norm(diff, dim=2)

    # k-nearest neighbors
    knn_dist, knn_idx = torch.topk(dist, k+1, dim=1, largest=False)
    knn_idx = knn_idx[:, 1:]  # remove self

    # Starting point: leftmost
    start = torch.argmin(points[:, 0])
    curr = start.item()

    # Initial direction (left)
    prev_dir = torch.tensor([-1.0, 0.0], device=device)

    hull = [curr]

    for _ in range(N * 2):
        neighbors = knn_idx[curr]
        vecs = points[neighbors] - points[curr]

        # angle with previous direction
        cosang = torch.sum(vecs * prev_dir, dim=1) / (
            torch.norm(vecs, dim=1) * torch.norm(prev_dir) + 1e-8
        )
        ang = torch.acos(torch.clamp(cosang, -0.9999, 0.9999))

        # pick smallest turning angle
        nxt = neighbors[torch.argmin(ang)].item()

        if nxt == start.item() and len(hull) > 3:
            break

        hull.append(nxt)

        # update marching direction
        new_dir = points[nxt] - points[curr]
        prev_dir = new_dir / (torch.norm(new_dir) + 1e-8)

        curr = nxt

    return points[hull]


def rasterize_polygon(points, H, W):
    """
    Rasterizes a polygon (concave or convex) into a binary mask on GPU.

    Parameters
    ----------
    points : (M, 2) torch tensor (x,y order), must be on CUDA
    H, W   : height and width of output mask

    Returns
    -------
    mask : (H, W) float32 tensor (0 or 1) on same device as points
    """
    device = points.device
    M = points.shape[0]

    # Close polygon if needed
    if not torch.all(points[0] == points[-1]):
        points = torch.cat([points, points[:1]], dim=0)

    x = points[:, 0]
    y = points[:, 1]

    # Grid of pixel centers
    ys = torch.linspace(0, H - 1, H, device=device)
    xs = torch.linspace(0, W - 1, W, device=device)

    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    X = X.T
    Y = Y.T

    # Edges of polygon
    x1 = x[:-1]
    y1 = y[:-1]
    x2 = x[1:]
    y2 = y[1:]

    # For each edge, compute intersections with scanlines
    cond1 = ((y1 <= Y[:,:,None]) & (y2 > Y[:,:,None])) \
          | ((y2 <= Y[:,:,None]) & (y1 > Y[:,:,None]))

    # Intersection point x coordinate
    t = (Y[:,:,None] - y1) / (y2 - y1 + 1e-12)
    x_intersect = x1 + t * (x2 - x1)

    # Count crossings to the right of pixel
    crossings = (x_intersect > X[:,:,None]) & cond1
    crossings = crossings.sum(dim=2)

    # Even–odd rule: inside if odd number of crossings
    mask = (crossings % 2 == 1).float()

    return mask


def binary_polygon_generator(size, n=100, concavity=0.3, allow_holes=True, smoothing=True, device=None):
    """Generates a binary image of a polygon with a concave hull."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    coords = sample_points_circle(n, size // 2, device=device)
    hull_points = concave_hull(coords, ratio=concavity)
    img = rasterize_polygon(hull_points, size, size).bool()
    
    if smoothing:
        min_blur = max(0.5, 2 * (size / 200))
        max_blur = min(10, 5 * (size / 200))
        blur_val = random.uniform(min_blur, max_blur)
        
        # Add batch/channel dimensions for conv2d
        img_tensor = img.float().unsqueeze(0).unsqueeze(0)
        
        # Create Gaussian kernel
        kernel_size = int(2 * math.ceil(3 * blur_val) + 1)
        sigma = blur_val
        x = torch.arange(kernel_size, device=device).float() - kernel_size // 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)
        
        # Apply Gaussian blur
        padding = kernel_size // 2
        img_blurred = F.conv2d(img_tensor, gauss_2d, padding=padding)
        
        # Threshold and return
        res = (img_blurred.squeeze() > 0.5).bool()
        return res
    else:
        return img
    
    
    
if __name__ == "__main__":
    import skimage.io as skio
    size = 400
    for i in range(200):
        n = random.randint(30,150)
        concavity = np.random.uniform(0.2,0.5)
        allow_holes = random.choice([True,False])
        smoothing = random.choice([True,False])
        binary_image = 1-binary_polygon_generator(size, n = n, concavity = concavity, allow_holes = allow_holes, smoothing = smoothing)

        binary_image = np.uint8(binary_image*255)
        skio.imsave(f'../polygons/polygon_{i}.png',binary_image)