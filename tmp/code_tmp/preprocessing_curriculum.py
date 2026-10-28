from cartoon.cartoon_utils import chambolle_tv_l1
import numpy as np
import os
import skimage.io as skio
from multiprocessing import Pool
from time import time, clock_gettime, CLOCK_MONOTONIC
import argparse


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

def cartoonize_multiple_levels(image_path):
    print(image_path)
    """Cartoonize an image at multiple levels of regularization.

    Args:
        image_path (str): Path to the input image.
        lambda_values (list): List of regularization parameters.

    Returns:
        list: List of cartoonized images.
    """
    lambda_values = [0.1,0.3,0.5,0.7,0.9,1.2,1.5,1.8] 
    target_directory = "/scratch/Raphael/data/DIV2K_cartoon/"
    cartoon_images = []
    for lambda_value in lambda_values:
        print(f"lambda_value :{lambda_value:.2f}")
        cartoon_image = cartoonize_image(image_path, lambda_value)
        new_filename = image_path.split("/")[-1][:-4]+f"_{lambda_value:.2f}.png"
        print(new_filename)
        skio.imsave(os.path.join(target_directory, new_filename), cartoon_image)
        cartoon_images.append(cartoon_image)
    return cartoon_images


if __name__ == """__main__""":
    
    parser = argparse.ArgumentParser(description="Building the training patch database")
    # Preprocessing parameters
    parser.add_argument("--start"           , type=int      , default=0    , help="Patch size")
    parser.add_argument("--end"             , type=int      , default=100  , help="Size of stride")
    parser.add_argument("--nb_process"      , type=int      , default=10   , help="Size of stride")
    
    
    args = parser.parse_args()
    root_dir = "/scratch/Raphael/data/DIV2K/DIV2K_train_HR"
    nb_p = args.nb_process
    files_list = [os.path.join(root_dir,f) for f in os.listdir(root_dir)][args.start:args.end]
    
    pool = Pool(nb_p)
    pool.map(cartoonize_multiple_levels,files_list)
    with Pool(10) as pool:
        res = [pool.apply_async(main) for _ in range(2)]
    # # pool.map(main,[_ for i in range(nb_p)])