#%%
from dead_leaves_generation import *
from dead_leaves_generation.utils.interpolation_maps import sample_period
from dead_leaves_generation.utils.texture_generation import bilevelTextureMixer
import matplotlib.pyplot as plt
# %%

texture = sample_period(0,10,10)
# %%
res = bilevelTextureMixer(single_color1=False,single_color2=True,mixing_types=["noise","sin"],width=500,warp=False)
# %%
plt.imshow(res)
plt.show()
# %%

from datasets import load_dataset
import numpy as np
from skimage.measure import label, regionprops
from random import sample

ds = load_dataset("nateraw/pascal-voc-2012")

# %%

import matplotlib.pyplot as plt 
plt.imshow(ds['train'][0]['image'])
plt.show()

# %%
plt.imshow(ds['train'][0]['mask'])
plt.show()
#%%
ds['train'][0].keys()
# %%

import numpy as np
from random import sample
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.draw import disk



def isolate_largest_object(ds, num_images=1000, output_size=256):
    binary_images = []
    indices = sample(range(len(ds['train'])), num_images)

    for i in indices:
        print(f"Processing image index: {i}")

        # --------------------------------------------------------
        # 1. Load mask and collapse to 1-channel
        # --------------------------------------------------------
        mask = np.array(ds['train'][i]['mask'])
        if mask.ndim == 3:
            mask = np.any(mask > 0, axis=-1).astype(np.uint8)

        H, W = mask.shape

        # --------------------------------------------------------
        # 2. Label components and pick largest
        # --------------------------------------------------------
        labeled_mask = label(mask)
        regions = regionprops(labeled_mask)
        if not regions:
            continue

        largest_region = max(regions, key=lambda r: r.area)
        if largest_region.solidity > 0.9:
            print (f"Skipping image index {i} due to high solidity: {largest_region.solidity}")  
            continue
        largest_binary = (labeled_mask == largest_region.label).astype(np.uint8)

        # --------------------------------------------------------
        # 3. Compute bounding box + add 1/8 margin
        # --------------------------------------------------------
        min_row, min_col, max_row, max_col = largest_region.bbox

        obj_h = max_row - min_row
        obj_w = max_col - min_col
        margin = int(max(obj_h, obj_w) / 8)

        min_row_m = max(0, min_row - margin)
        min_col_m = max(0, min_col - margin)
        max_row_m = min(H, max_row + margin)
        max_col_m = min(W, max_col + margin)

        cropped = largest_binary[min_row_m:max_row_m, min_col_m:max_col_m]

        # --------------------------------------------------------
        # 4. Resize with preserved aspect ratio (no squishing)
        # --------------------------------------------------------
        ch, cw = cropped.shape

        # Scale factor: longest side → output_size
        scale = output_size / max(ch, cw)

        new_h = int(ch * scale)
        new_w = int(cw * scale)

        scaled = resize(
            cropped,
            (new_h, new_w),
            order=0,
            preserve_range=True,
            anti_aliasing=False
        ).astype(np.uint8)

        # --------------------------------------------------------
        # 5. Pad to square (centered)
        # --------------------------------------------------------
        pad_h = output_size - new_h
        pad_w = output_size - new_w

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        final = np.pad(
            scaled,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=0
        )

        binary_images.append(final)

    return binary_images

binary_images = isolate_largest_object(ds)
# %%

import skimage.io as skio

for i in range(200):
    img = (255-255*binary_images[i]).astype(np.uint8)
    skio.imsave(f'pascal_voc/object_{i}.png',img)
# %%
from skimage.draw import disk
def generate_binary_disk(radius=50, image_size=256):
    center = (image_size // 2, image_size // 2)
    binary_image = np.zeros((image_size, image_size), dtype=np.uint8)
    rr, cc = disk(center, radius, shape=binary_image.shape)
    binary_image[rr, cc] = 1
    return binary_image

binary_disk = 1- generate_binary_disk(radius = 100)

skio.imsave('binary_disk.png', np.uint8(binary_disk*255))
# plt.imshow(binary_disk, cmap='gray')
# plt.show()
# %%
