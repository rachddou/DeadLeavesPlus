import numpy as np
import matplotlib.pyplot as plt
import numba
import skimage.io as skio
from numba import jit
from cartoon_utils import chambolle_tv_l1
from time import time


def main(image_path,lambda_value=0.5):
    image = skio.imread(image_path)/255.
    print(f"Image shape: {image.shape}")
    h,w,c = image.shape
    image_size = h*w

    cartoon_image = np.zeros((h, w, c), dtype=np.float32)
    
    nb_iter_max = 1000
    tau = 0.35
    sigma = 0.35
    theta = 1
    
    for i in range(c):
        channel = image[:, :, i]

        cartoon_channel = chambolle_tv_l1(channel, nb_iter_max, tau, sigma, lambda_value, theta)
        cartoon_image[:, :, i] = cartoon_channel

    cartoon_image = np.clip(cartoon_image, 0, 1)  # Ensure pixel values are in [0, 1]
    cartoon_image = (cartoon_image * 255).astype(np.uint8)  #
    return cartoon_image


if __name__ == "__main__":
    image_path = '/scratch/Raphael/data/Set14/image_SRF_2/HR/img_002_SRF_2_HR.png'  # Replace with your image path
    lambda_value = 0.3  # Adjust as needed
    t0 = time()
    cartoon_image = main(image_path, lambda_value)
    t1 = time()

    print(f"Cartoonization completed in {t1 - t0:.2f} seconds")
    plt.imshow(cartoon_image)
    plt.axis('off')
    plt.show()  # Display the cartoonized image