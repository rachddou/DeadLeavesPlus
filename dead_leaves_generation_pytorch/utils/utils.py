import torch
import torch.nn.functional as F
import numpy as np
from dead_leaves_generation_pytorch.utils.color_conversion import rgb2lab_torch, lab2rgb_torch


## rotation function using OpenCV
def rotate_CV(image, angle, device=None):
    """
    Rotate an image by a given angle using PyTorch.
    
    Args:
        image: torch.Tensor of shape (H, W) or (H, W, C)
        angle: float, angle of rotation in degrees
        device: torch device
    
    Returns:
        torch.Tensor: rotated image
    """
    if device is None:
        device = image.device if isinstance(image, torch.Tensor) else torch.device('cpu')
    
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).to(device)
    
    angle_rad = torch.tensor(angle * np.pi / 180.0, device=device)
    
    # Handle different dimensions
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
        squeeze_output = True
    else:
        image = image.permute(2, 0, 1).unsqueeze(0)
        squeeze_output = False
    
    # Create rotation matrix
    theta = torch.zeros(1, 2, 3, device=device)
    theta[0, 0, 0] = torch.cos(angle_rad)
    theta[0, 0, 1] = torch.sin(angle_rad)
    theta[0, 1, 0] = -torch.sin(angle_rad)
    theta[0, 1, 1] = torch.cos(angle_rad)
    
    grid = F.affine_grid(theta, image.size(), align_corners=False)
    rotated = F.grid_sample(image.float(), grid, align_corners=False, mode='bilinear')
    
    if squeeze_output:
        return rotated.squeeze()
    else:
        return rotated.squeeze(0).permute(1, 2, 0)


# normalizing/ thresholding function
def normalize(x):
    """
    Normalize a tensor between 0 and 1.
    
    Args:
        x: torch.Tensor
    
    Returns:
        torch.Tensor: normalized tensor
    """
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x

def thresholding(img, a, b):
    """
    Soft threshold with min=a and max=b.
    
    Args:
        img: torch.Tensor
        a: lower threshold
        b: upper threshold
    
    Returns:
        torch.Tensor: thresholded image
    """
    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img)
    img = torch.clamp(img, a, b)
    img = ((img - a) / (b - a) * 255).byte()
    return img


## functions for counting remaining disks
def theoretical_number_disks(delta, r_min, r_max, width, alpha):
    """function that computes the theoretical number of disks in a dead leaves image with given parameters.

    Args:
        delta (float): percentage of the image covered by the disks
        r_min (int): minimal radius of the disks
        r_max (int): maximal radius of the disks
        width (int): width of the image
        alpha (float): power law parameter 
    """
    d = 0
    e =  alpha - 1
    for k in range(r_min,r_max,1):
        d+=(1/((1/r_min**e) -(1/r_max**e))*(1-((k/(k+1))**2)))
    d*=(np.pi)/(width**2)
    if d>=1.0:
        return(20)
    else:
        return int(np.log(delta) / np.log(1-d))


def N_sup_r(r, width, r_min, r_max):
    delta = 100 / (width**2)
    N = theoretical_number_disks(delta, r_min, r_max, width, 3)
    prop = ((1/r**2) - (1/r_max**2)) / (1/(r_min**2) - 1/(r_max**2))
    res = N * prop
    return res


## logit functions
def logistic(x, L=1, x_0=0, k=1):
    """Logistic function using PyTorch."""
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)
    return L / (1 + torch.exp(-k * (x - x_0)))

def normalized_logistic(x, k, device=None):
    """Normalized logistic function."""
    if device is None:
        device = x.device if isinstance(x, torch.Tensor) else torch.device('cpu')
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).to(device)
    
    y = (logistic(12.1*x - 6, k=k) - logistic(torch.tensor(-6.0, device=device), k=k)) / \
        (logistic(torch.tensor(6.1, device=device), k=k) - logistic(torch.tensor(-6.0, device=device), k=k))
    return y


def sigmoid(x, lamda, device=None):
    """Sigmoid function to transform sinusoid into sharper transition."""
    if device is None:
        device = x.device if isinstance(x, torch.Tensor) else torch.device('cpu')
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).to(device)
    
    sig = 1 / (1 + torch.exp(-lamda * x))
    if lamda >= 20:
        sig = (sig > 0.5).float()
    return sig


## color gradient function

def linear_color_gradient(color_1, color_2, width, angle, k=0.5, color_space="lab", device=None):
    """
    Create color gradient between two colors using PyTorch.
    
    Args:
        color_1: first color in RGB (torch.Tensor or numpy array) with values in [0, 255]
        color_2: second color in RGB with values in [0, 255]
        width: width of gradient
        angle: angle of gradient
        k: smoothness factor
        color_space: "rgb" or "lab"
        device: torch device
    
    Returns:
        torch.Tensor: gradient image (uint8)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create linear gradient
    lin_gradient = torch.linspace(0, 1, 2*width, device=device).unsqueeze(0).repeat(2*width, 1)
    rotated_grad = normalize(rotate_CV(lin_gradient, angle, device=device))
    
    # Ensure 2D tensor
    if rotated_grad.dim() > 2:
        rotated_grad = rotated_grad.squeeze()
    
    h, w = rotated_grad.shape
    if h >= width and w >= width:
        rotated_grad = rotated_grad[h//2-width//2:h//2+width//2, w//2-width//2:w//2+width//2]
    
    lin_gradient = normalized_logistic(rotated_grad, k=k, device=device)
    
    # Convert colors to tensors
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
    
    # Convert to color space
    if color_space == "lab":
        color_transform_1 = rgb2lab_torch(color_1, device=device)
        color_transform_2 = rgb2lab_torch(color_2, device=device)
    else:
        color_transform_1 = color_1
        color_transform_2 = color_2
    
    # Interpolate in color space
    x = color_transform_1[0] + (color_transform_2[0] - color_transform_1[0]) * lin_gradient
    y = color_transform_1[1] + (color_transform_2[1] - color_transform_1[1]) * lin_gradient
    z = color_transform_1[2] + (color_transform_2[2] - color_transform_1[2]) * lin_gradient
    
    final_img = torch.stack([x, y, z], dim=-1)
    
    # Convert back to RGB if in LAB space
    if color_space == "lab":
        final_img = lab2rgb_torch(final_img, device=device)
        final_img = (final_img * 255.0).clamp(0, 255).byte()
    else:
        final_img = (final_img * 255.0).clamp(0, 255).byte()
    
    return final_img