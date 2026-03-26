import logging
import os
import random
from pathlib import Path
from time import time

import cv2
import numpy as np
import skimage.io as skio
from blurgenerator import lens_blur

from dead_leaves_generation.polygons_maker_bis import binary_polygon_generator, make_rectangle_mask
from dead_leaves_generation.utils.texture_generation import bilevelTextureMixer, pattern_patch_two_colors
from dead_leaves_generation.utils.utils import theoretical_number_disks, linear_color_gradient
from dead_leaves_generation.utils.colored_noise import sample_color_noise
from dead_leaves_generation.utils.perspective import perspective_shift

logger = logging.getLogger(__name__)

_NPY_PATH = Path(__file__).parent.parent / "npy" / "dict.npy"
dict_instance = np.load(_NPY_PATH, allow_pickle=True)


class Textures:
    def __init__(self, width=1000, use_natural_images=True, path="",
                 types=None, type_weights=None,
                 slope_range=None,
                 img_source=None,
                 apply_warp=True, random_phase=False):
        if types is None:
            types = ["sin"]
        if type_weights is None:
            type_weights = [1]
        if slope_range is None:
            slope_range = [0.5, 2.5]
        if img_source is None:
            img_source = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

        self.width = width
        self.use_natural_images = use_natural_images
        self.path = path
        self.apply_warp = apply_warp
        self.random_phase = random_phase
        self.online_generation = True

        self.img_source = img_source
        self.src_h = img_source.shape[0]
        self.src_w = img_source.shape[1]
        self.files = []
        self.resulting_image = np.ones((width, width, 3), dtype=np.uint8)
        self.perspective_shift = False
        self.perspective_var = True

        if self.use_natural_images:
            self.source_image_sampling()
        self.texture_type_lists = types
        self.texture_type_frequency = type_weights
        self.slope_range = slope_range
        self.texture_type = types[0]
        self.sample_texture_type()

    def _sample_slope_from_ranges(self):
        """Samples a slope value uniformly from a union of disjoint intervals.

        Returns:
            float: The sampled slope value.
        """
        if not isinstance(self.slope_range[0], (int, float)):
            # Multiple intervals: [[a1,b1], [a2,b2], ...]
            intervals = self.slope_range
            lengths = [interval[1] - interval[0] for interval in intervals]
            total_length = sum(lengths)
            interval_probs = [length / total_length for length in lengths]
            chosen_interval = np.random.choice(len(intervals), p=interval_probs)
            interval = intervals[chosen_interval]
            return np.random.uniform(interval[0], interval[1])
        else:
            # Single interval: [a, b]
            return np.random.uniform(self.slope_range[0], self.slope_range[1])

    def lin_gradient(self, color1, color2, angle=45):
        """Generates a linear color gradient between two colors.

        Args:
            color1 (tuple): The first color (R, G, B).
            color2 (tuple): The second color (R, G, B).
            angle (int, optional): The angle of the gradient. Defaults to 45.
        """
        k = np.random.uniform(0.1, 0.5)
        return linear_color_gradient(color_1=color1, color_2=color2,
                                     width=self.width, angle=angle,
                                     k=k, color_space="lab")

    def random_patch_selection(self):
        """Selects a random 100x100 patch from the source image.

        Returns:
            np.ndarray: The selected patch (100 x 100 x 3).
        """
        x = np.random.randint(0, max(1, self.src_h - 101))
        y = np.random.randint(0, max(1, self.src_w - 101))
        return self.img_source[x:x + 100, y:y + 100]

    def generate_source_texture(self, width=1000):
        """Generates a texture map from color noise or a bilevel texture mixer.

        Args:
            width (int, optional): Size of the texture map. Defaults to 1000.
        """
        if self.texture_type == "freq_noise":
            slope = self._sample_slope_from_ranges()
            return sample_color_noise(self.random_patch_selection(), width=width, slope=slope)

        if self.texture_type == "texture_mixes":
            # 90% sin, 10% grid
            singleColor1 = np.random.random() < 0.1
            singleColor2 = np.random.random() < 0.1
            mix_modes = [["sin"], ["grid"]]
            textureMixMode = mix_modes[np.random.choice(2, p=[0.9, 0.1])]
        else:
            # texture_type in ["sin", "grid"]
            singleColor1 = True
            singleColor2 = True
            textureMixMode = [self.texture_type]

        warp = np.random.random() < 0.5 if self.apply_warp else False
        thresh = np.random.randint(5, 50)
        return bilevelTextureMixer(
            color_source_1=self.random_patch_selection(),
            color_source_2=self.random_patch_selection(),
            single_color1=singleColor1, single_color2=singleColor2,
            mixing_types=textureMixMode, width=width,
            thresh_val=thresh, warp=warp,
            slope_range=self.slope_range)

    def source_image_sampling(self):
        """Selects a random source image from the path directory.
        Updates img_source, src_h and src_w.
        """
        if not self.files:
            self.files = [os.path.join(self.path, f)
                          for f in os.listdir(self.path)
                          if os.path.isfile(os.path.join(self.path, f))]
        ind = random.randint(0, len(self.files) - 1)
        self.img_source = skio.imread(self.files[ind])
        self.src_h = self.img_source.shape[0]
        self.src_w = self.img_source.shape[1]
        logger.debug("Sampling source image: %s", self.files[ind])

    def sample_texture_type(self):
        """Samples a texture type from the available list."""
        if len(self.texture_type_lists) == 1:
            self.texture_type = self.texture_type_lists[0]
        else:
            self.texture_type = np.random.choice(self.texture_type_lists,
                                                  p=self.texture_type_frequency)

    def generate_texture(self, width=100):
        """Generates a texture map based on the selected texture type.

        Args:
            width (int, optional): The width of the texture map. Defaults to 100.

        Returns:
            np.ndarray: The generated texture map (width x width x 3, uint8).
        """
        self.sample_texture_type()
        tmp_random_phase = self.random_phase
        logger.debug("Generating texture of type: %s", self.texture_type)

        if self.texture_type == "gradient":
            color1 = np.uint8(self.img_source[np.random.randint(0, self.src_h),
                                              np.random.randint(0, self.src_w), :])
            color2 = np.uint8(self.img_source[np.random.randint(0, self.src_h),
                                              np.random.randint(0, self.src_w), :])
            return self.lin_gradient(color1, color2, angle=45)

        if self.texture_type in ["grid", "texture_mixes"] and self.random_phase:
            self.random_phase = False
        self.perspective_var = np.random.random() > 0.5
        if self.perspective_var and self.perspective_shift:
            logger.debug("Applying perspective shift")
            res = self.generate_source_texture(width=2 * width)
            res = perspective_shift(res)
        else:
            res = self.generate_source_texture(width=width)
        self.random_phase = tmp_random_phase
        return res

    def clear(self):
        self.resulting_image = np.ones((self.width, self.width, 3), dtype=np.uint8)


class Deadleaves(Textures):
    def __init__(self, rmin=1, rmax=1000, power_law_exponent=3, width=1000,
                 use_natural_images=True, path="",
                 texture_path="", shape_type="poly",
                 types=None, type_weights=None,
                 slope_range=None, use_texture=True,
                 online_generation=False, apply_warp=True,
                 random_phase=False, perspective=True,
                 img_source=None):
        # Deadleaves initialises all Textures attributes manually below to avoid
        # triggering the auto source_image_sampling() call inside Textures.__init__,
        # which is deferred here to compose_dead_leaves_depth_of_field().
        if types is None:
            types = ["sin"]
        if type_weights is None:
            type_weights = [1]
        if slope_range is None:
            slope_range = [0.5, 2.5]
        if img_source is None:
            img_source = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

        self.rmin = rmin
        self.rmax = rmax
        self.power_law_exponent = power_law_exponent
        self.width = width
        self.use_natural_images = use_natural_images
        self.path = path
        self.apply_warp = apply_warp
        self.random_phase = random_phase
        self.texture_path = texture_path
        self.shape_type = shape_type
        self.use_texture = use_texture
        self.texture_type_lists = types
        self.texture_type_frequency = type_weights
        self.slope_range = slope_range
        self.texture_type = types[0]
        self.sample_texture_type()
        self.perspective_shift = perspective
        self.perspective_var = True

        self.files = []
        self.img_source = img_source
        self.src_h = self.img_source.shape[0]
        self.src_w = self.img_source.shape[1]

        self.online_generation = online_generation
        self.n_textures = 10
        self.textures = []

        self.binary_image = np.ones((width, width), dtype=bool)
        self.resulting_image = np.ones((width, width, 3), dtype=np.uint8)

        self.vamin = 1 / (rmax ** (power_law_exponent - 1))
        self.vamax = 1 / (rmin ** (power_law_exponent - 1))
        theoretical_n_shapes = theoretical_number_disks(
            0.002, rmin, rmax, width, power_law_exponent)
        interval = int(theoretical_n_shapes)
        logger.debug("Expected shapes per layer: %d", interval)
        if interval < 10:
            interval = 1
        if interval > 10:
            interval = interval // 10
        self.interval = interval

    def update_leaves_size_parameters(self, rmin, rmax):
        """Updates the leaf size parameters and recomputes the shape interval.

        Args:
            rmin (float): The minimum shape radius.
            rmax (float): The maximum shape radius.
        """
        self.rmin = rmin
        self.rmax = rmax
        self.vamin = 1 / (self.rmax ** (self.power_law_exponent - 1))
        self.vamax = 1 / (self.rmin ** (self.power_law_exponent - 1))
        theoretical_n_shapes = theoretical_number_disks(
            0.002, self.rmin, self.rmax, self.width, self.power_law_exponent)
        interval = int(theoretical_n_shapes)
        logger.debug("Expected shapes per layer: %d", interval)
        if interval < 10:
            interval = 1
        if interval > 10:
            interval = interval // 10
        self.interval = interval

    def fetch_textures(self):
        """Loads n_textures textures from texture_path."""
        files = [os.path.join(self.texture_path, f)
                 for f in os.listdir(self.texture_path)]
        file_id = np.random.choice(len(files), self.n_textures)
        self.textures = [skio.imread(files[ind]) for ind in file_id]
        return self.textures

    def random_patch_selection(self):
        """Selects a random 100x100 patch from the source image.

        Returns:
            np.ndarray: A random patch (100 x 100 x 3).
        """
        x = np.random.randint(0, max(1, self.src_h - 101))
        y = np.random.randint(0, max(1, self.src_w - 101))
        return self.img_source[x:x + 100, y:y + 100, :]

    def generate_textures_dictionary(self):
        """Precomputes n_textures textures at 5 downscale levels."""
        textures = []
        for _ in range(self.n_textures):
            texture_img = self.generate_texture(width=2 * self.rmax + 100)
            scale_dict = {"1": texture_img}
            for i in range(2, 6):
                scale_dict[str(i)] = cv2.resize(
                    texture_img, (0, 0), fx=1. / i, fy=1. / i,
                    interpolation=cv2.INTER_AREA)
            textures.append(scale_dict)
        self.textures = textures

    def pick_texture(self, size):
        """Picks a texture from the precomputed dictionary.

        Args:
            size (int): The target size of the shape being filled.

        Returns:
            np.ndarray: A randomly transformed texture patch.
        """
        current_texture_dict = self.textures[np.random.randint(0, self.n_textures)]
        h = current_texture_dict["1"].shape[0]
        max_scale = min(5, h / size)
        scale = np.floor(1 + max(0, (max_scale - 1) * np.random.power(2 / 3))).astype(int)
        current_texture = current_texture_dict[str(scale)]

        transform = np.random.choice(4)
        if transform == 0:
            current_texture = np.flipud(current_texture)
        elif transform == 1:
            current_texture = np.fliplr(current_texture)
        elif transform == 2:
            current_texture = np.rot90(current_texture, axes=(0, 1))
        elif transform == 3:
            current_texture = np.flipud(np.rot90(current_texture, axes=(0, 1)))
        return current_texture

    def resize_textures(self, size, texture):
        """Resizes the given texture proportionally to match a target size.

        Args:
            size (int): The desired target size.
            texture (np.ndarray): The texture to resize.

        Returns:
            np.ndarray: The resized texture.
        """
        h = texture.shape[0]
        max_scale = min(5, h / size)
        scale = int((1 + (max_scale - 1) * np.random.power(2 / 3)) * 5) / 5.
        if scale > 1:
            texture = cv2.resize(texture, (0, 0), fx=1. / scale, fy=1. / scale,
                                 interpolation=cv2.INTER_AREA)
        return texture

    def generate_single_shape_mask(self):
        """Samples a shape radius from the power-law distribution and generates
        the corresponding binary mask.

        Returns:
            tuple: (shape_mask np.ndarray, radius int)
        """
        radius = self.vamin + (self.vamax - self.vamin) * np.random.random()
        radius = int(1 / (radius ** (1. / (self.power_law_exponent - 1))))
        shape = self.shape_type
        if shape == "mix":
            shape = random.choice(["disk", "poly", "rectangle"])

        if shape == "poly":
            shape_1d = binary_polygon_generator(
                2 * (3 * radius // 2) + 1,
                n=np.random.randint(50, max(100, int(0.9 * radius))),
                allow_holes=bool(random.getrandbits(1)),
                smoothing=bool(random.getrandbits(1)))
            if max(shape_1d.shape) >= 2 * self.rmax + 100:
                scale = (2 * self.rmax + 100.) / max(shape_1d.shape)
                new_size = (int(2 * ((shape_1d.shape[0] * scale) // 2) - 1),
                            int(2 * ((shape_1d.shape[1] * scale) // 2) - 1))
                shape_1d = np.bool_(cv2.resize(np.uint8(shape_1d), new_size,
                                               interpolation=cv2.INTER_AREA))
        elif shape == "disk":
            shape_1d = dict_instance[()][str(radius)]
        elif shape == "rectangle":
            shape_1d = make_rectangle_mask(radius)
            if max(shape_1d.shape) >= 2 * self.rmax + 100:
                scale = (2 * self.rmax + 100.) / max(shape_1d.shape)
                new_size = (int(2 * ((shape_1d.shape[0] * scale) // 2) - 1),
                            int(2 * ((shape_1d.shape[1] * scale) // 2) - 1))
                shape_1d = np.bool_(cv2.resize(np.uint8(shape_1d), new_size,
                                               interpolation=cv2.INTER_AREA))
        return shape_1d, radius

    def add_shape_to_binary_mask(self, shape_1d):
        """Places a shape at a random position and updates the occlusion mask.

        Args:
            shape_1d (np.ndarray): Binary shape mask.

        Returns:
            tuple: (x_min, x_max, y_min, y_max, shape_mask_1d)
        """
        width_shape, length_shape = shape_1d.shape[0], shape_1d.shape[1]
        pos = [np.random.randint(0, self.width), np.random.randint(0, self.width)]

        x_min = max(0, pos[0] - width_shape // 2)
        x_max = min(self.width, 1 + pos[0] + width_shape // 2)
        y_min = max(0, pos[1] - length_shape // 2)
        y_max = min(self.width, 1 + pos[1] + length_shape // 2)

        shape_mask_1d = self.binary_image[x_min:x_max, y_min:y_max].copy()
        shape_1d = shape_1d[
            max(0, width_shape // 2 - pos[0]):
                min(width_shape, self.width + width_shape // 2 - pos[0]),
            max(0, length_shape // 2 - pos[1]):
                min(length_shape, self.width + length_shape // 2 - pos[1])]
        shape_mask_1d *= shape_1d
        self.binary_image[x_min:x_max, y_min:y_max] *= np.logical_not(shape_mask_1d)
        return x_min, x_max, y_min, y_max, shape_mask_1d

    def render_shape(self, shape_mask_1d, radius):
        """Fills a shape mask with a texture or colour.

        Args:
            shape_mask_1d (np.ndarray): Binary mask of the shape.
            radius (int): Radius of the shape (used to decide fill type).

        Returns:
            tuple: (shape_mask float32, shape_render uint8)
        """
        width_shape, length_shape = shape_mask_1d.shape[0], shape_mask_1d.shape[1]
        shape_mask = np.float32(np.repeat(shape_mask_1d[:, :, np.newaxis], 3, axis=2))
        shape_render = shape_mask.copy()
        color = np.uint8(self.img_source[np.random.randint(0, self.src_h),
                                         np.random.randint(0, self.src_w), :])
        if radius < 20:
            shape_render = color * shape_render
        else:
            angle = np.random.randint(0, 360)
            color_2 = np.uint8(self.img_source[np.random.randint(0, self.src_h),
                                               np.random.randint(0, self.src_w), :])
            grad_vs_texture = np.random.random() if self.use_texture else 1.0
            if grad_vs_texture > 0.95:
                k = np.random.uniform(0.1, 0.5)
                textureMap = linear_color_gradient(
                    color, color_2,
                    width=2 * max(width_shape, length_shape) + 1,
                    angle=angle, k=k, color_space="lab")
            else:
                if self.online_generation:
                    textureMap = self.generate_texture(width=60 + max(width_shape, length_shape))
                else:
                    textureMap = self.pick_texture(size=max(width_shape, length_shape))

            h, w = textureMap.shape[0], textureMap.shape[1]
            x = np.random.randint(0, max(1, h - width_shape))
            y = np.random.randint(0, max(1, w - length_shape))
            textureMap = textureMap[x:x + width_shape, y:y + length_shape]
            shape_render = np.uint8(np.float32(shape_render) * textureMap)

        return shape_mask, shape_render

    def generate_stack(self, disk_count):
        """Generates a stack of rendered shapes.

        Args:
            disk_count (int): Number of shapes to render.

        Returns:
            tuple: (resulting_image uint8, binary_mask uint8)
        """
        for _ in range(disk_count):
            shape_1d, radius = self.generate_single_shape_mask()
            x_min, x_max, y_min, y_max, shape_mask_1d = self.add_shape_to_binary_mask(shape_1d)
            shape_mask, shape_render = self.render_shape(shape_mask_1d, radius)
            self.resulting_image[x_min:x_max, y_min:y_max, :] *= np.uint8(1 - shape_mask)
            self.resulting_image[x_min:x_max, y_min:y_max, :] += np.uint8(shape_render)
        logger.debug("Dead leaves stack created (%d shapes)", disk_count)
        return (self.resulting_image.copy(),
                np.uint8(1 - np.repeat(self.binary_image[..., np.newaxis], 3, axis=2)).copy())

    def clear(self):
        self.resulting_image = np.ones((self.width, self.width, 3), dtype=np.uint8)
        self.binary_image = np.ones((self.width, self.width), dtype=bool)

    def source_image_sampling(self):
        """Selects a random source image from the path directory.
        Updates img_source, src_h and src_w.
        """
        if not self.files:
            self.files = [os.path.join(self.path, f)
                          for f in os.listdir(self.path)
                          if os.path.isfile(os.path.join(self.path, f))]
        ind = random.randint(0, len(self.files) - 1)
        self.img_source = skio.imread(self.files[ind])
        self.src_h = self.img_source.shape[0]
        self.src_w = self.img_source.shape[1]
        logger.debug("Sampling source image: %s", self.files[ind])

    def compose_dead_leaves_depth_of_field(self, blur_type, blur_val, fetch=False):
        """Composes the dead leaves image with a three-layer depth-of-field effect.

        Args:
            blur_type (str): Type of blur to apply ("gaussian" or "lens").
            blur_val (float): Blur strength (sigma for Gaussian, radius for lens).
            fetch (bool, optional): Load textures from texture_path instead of
                generating them. Defaults to False.
        """
        if self.use_natural_images:
            for _ in range(10):
                self.source_image_sampling()
                if self.img_source.ndim == 3 and self.img_source.shape[2] == 3:
                    break
            else:
                raise RuntimeError(
                    "Could not load a 3-channel source image after 10 attempts.")

        if self.use_texture:
            if not self.online_generation:
                if fetch:
                    self.fetch_textures()
                else:
                    self.generate_textures_dictionary()

        background, _ = self.generate_stack(10 * self.interval)
        self.clear()
        plainground, plain_mask = self.generate_stack(self.interval)
        self.clear()
        foreground, fore_mask = self.generate_stack(int(0.5 * self.interval))

        # Blur background, composite with in-focus plainground
        if blur_type == "gaussian":
            background = cv2.GaussianBlur(background, (11, 11), sigmaX=blur_val,
                                          borderType=cv2.BORDER_DEFAULT)
        elif blur_type == "lens":
            background = lens_blur(background, radius=blur_val, components=2, exposure_gamma=2)
        plainground = (1 - plain_mask) * background + plain_mask * plainground

        # Blur foreground, composite over plainground
        foreground += (1 - fore_mask) * plainground
        if blur_type == "gaussian":
            foreground = cv2.GaussianBlur(foreground, (11, 11), sigmaX=blur_val,
                                          borderType=cv2.BORDER_DEFAULT)
            fore_mask = cv2.GaussianBlur(fore_mask, (11, 11), sigmaX=blur_val,
                                         borderType=cv2.BORDER_DEFAULT)
        elif blur_type == "lens":
            foreground = lens_blur(foreground, radius=blur_val, components=2, exposure_gamma=2)
            fore_mask = lens_blur(255 * fore_mask, radius=blur_val, components=2,
                                  exposure_gamma=2) / 255.

        resulting_img = plainground * (1 - fore_mask) + foreground * fore_mask
        self.resulting_image = np.clip(resulting_img, 0, 255)

    def postprocess(self, blur=True, downscale=True):
        """Post-processes the resulting image.

        Args:
            blur (bool, optional): Apply an additional Gaussian blur. Defaults to True.
            downscale (bool, optional): 2x downsample the image. Defaults to True.
        """
        if blur or downscale:
            if blur:
                blur_value = np.random.uniform(1, 3)
                self.resulting_image = cv2.GaussianBlur(
                    self.resulting_image, (11, 11),
                    sigmaX=blur_value, borderType=cv2.BORDER_DEFAULT)
            if downscale:
                self.resulting_image = cv2.resize(
                    self.resulting_image, (0, 0),
                    fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            self.resulting_image = np.uint8(self.resulting_image)
