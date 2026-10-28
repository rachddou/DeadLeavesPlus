from dead_leaves_generation.utils.texture_generation import bilevelTextureMixer
from dead_leaves_generation.utils.colored_noise import sample_color_noise
import matplotlib.pyplot as plt
import skimage.io as skio
import os
import numpy as np

path = "/Users/raphael/Workspace/telecom/code/exploration_database_and_code/pristine_images/"
f = os.listdir(path)
n = len(f)

curr_f = np.random.choice(n-1)
image = skio.imread(path+f[curr_f])[0:100,0:100,:]
width = 300
slopes = np.arange(0.5,2.5,0.33)
for i in range(slopes.size):
    slope = slopes[i]
    yo  = sample_color_noise(image,width,slope)
    # yo  = bilevelTextureMixer(single_color1=True,single_color2=True,mixing_types=['sin'],width=300,warp=False)
    skio.imsave("texture_slope_{}.png".format(str(slope).zfill(2)), yo)
