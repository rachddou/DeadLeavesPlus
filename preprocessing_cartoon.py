from cartoon.cartoon_utils import chambolle_tv_l1
import numpy as np
import os
import skimage.io as skio
from multiprocessing import Pool
from time import time, clock_gettime, CLOCK_MONOTONIC


def cartoonize_image(image_path, lambda_value=0.5):
    """Cartoonize an image using the Chambolle TV L1 method.

    Args:
        image_path (str): Path to the input image.
        lambda_value (float): Regularization parameter for the Chambolle TV L1 method.

    Returns:
        np.ndarray: Cartoonized image.
    """
    image = skio.imread(image_path) / 255.0
    print(f"Image shape: {image.shape}")
    
    cartoon_image = np.zeros_like(image, dtype=np.float32)
    
    nb_iter_max = 1000
    tau = 0.35
    sigma = 0.35
    theta = 1
    
    for i in range(image.shape[2]):
        channel = image[:, :, i]
        cartoon_channel = chambolle_tv_l1(channel, nb_iter_max, tau, sigma, lambda_value, theta)
        cartoon_image[:, :, i] = cartoon_channel

    cartoon_image = np.clip(cartoon_image, 0, 1) * 255
    return cartoon_image.astype(np.uint8)

def cartoonize_multiple_levels(image_path, lambda_values, target_directory):
    """Cartoonize an image at multiple levels of regularization.

    Args:
        image_path (str): Path to the input image.
        lambda_values (list): List of regularization parameters.

    Returns:
        list: List of cartoonized images.
    """
    cartoon_images = []
    for lambda_value in lambda_values:
        cartoon_image = cartoonize_image(image_path, lambda_value)
        new_filename = f"cartoonized_lambda_{lambda_value:.2f}.png"
        skio.imsave(os.path.join(target_directory, new_filename), cartoon_image)
        cartoon_images.append(cartoon_image)
    return cartoon_images