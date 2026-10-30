import logging
import os
import random
from functools import lru_cache

import cv2
import numpy as np
import skimage.io as skio
from blurgenerator import lens_blur

from dead_leaves_generation.polygons_maker_bis import binary_polygon_generator, make_rectangle_mask
from dead_leaves_generation.utils.texture_generation import bilevel_texture_mixer, sample_slope
from dead_leaves_generation.utils.utils import (
    linear_color_gradient, number_of_leaves, theoretical_number_disks)
from dead_leaves_generation.utils.colored_noise import sample_color_noise
from dead_leaves_generation.utils.perspective import perspective_shift, perspective_source_box
from dead_leaves_generation.utils.kernels import seed_numba

logger = logging.getLogger(__name__)

# True: number of shapes per layer as in the original implementation (e.g. to
# compare images with it), which only covers the image for a power-law exponent
# of 3 and leaves its borders partly uncovered. False: 99.9% of the image is
# covered after the three layers, see number_of_leaves.
EXACT_ORIGINAL = False

# Mean area of the shape masks, as (radii, factors), measured over 40-1500 masks
# per radius (linear in log radius in between, constant beyond):
# - polygons: relative to pi * (3 * radius // 2)^2, the circle in which their
#   points are drawn; they fill more of it as the number of points (0.9 *
#   radius above radius 111) grows;
# - rectangles: relative to pi * radius^2 (their target area, minus rounding).
SHAPE_AREA_FACTORS = {
    "poly": ((1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 60, 100, 125, 150, 200, 250, 300, 400,
              500, 650, 800, 1000),
             (0.318, 0.246, 0.308, 0.335, 0.343, 0.362, 0.376, 0.384, 0.394, 0.401, 0.408, 0.412,
              0.416, 0.413, 0.44, 0.451, 0.483, 0.52, 0.532, 0.562, 0.61, 0.615, 0.704, 0.7)),
    "rectangle": ((1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30),
                  (0.318, 0.249, 0.58, 0.723, 0.841, 0.887, 0.925, 0.941, 0.955, 0.979, 0.988,
                   0.998, 1.0)),
}
UNCOVERED_FRACTION = 1e-3

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def list_images(path):
    """Sorted paths of the image files of a directory ([] if it does not exist).

    Sorted: os.listdir's order depends on the file system, and the same random
    draw must pick the same image everywhere.
    """
    if not os.path.isdir(path):
        return []
    return sorted(os.path.join(path, f) for f in os.listdir(path)
                  if f.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(path, f)))


def _mix64(x):
    """SplitMix64 finaliser: decorrelated 64-bit keys from consecutive integers."""
    x &= 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


@lru_cache(maxsize=256)
def disk_mask(radius):
    """Binary disk of the given radius: pixels (i, j) with i^2 + j^2 <= radius^2
    around the centre, as a (2 * radius + 1)^2 read-only array."""
    L = np.arange(-radius, radius + 1, dtype=np.int32)
    X, Y = np.meshgrid(L, L)
    mask = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=bool)
    mask.setflags(write=False)
    return mask


class Textures:
    """Generates random textures whose colours are sampled from a source image.

    Args:
        width (int): Default texture size.
        use_natural_images (bool): Sample colours from the images in ``path``
            (otherwise from ``img_source``).
        path (str): Directory of colour source images.
        types (list): Texture types to sample from, among "sin", "grid",
            "freq_noise", "texture_mixes" and "gradient".
        type_weights (list): Sampling probability of each type.
        slope_range: 1/f^slope range of the coloured noise, [a, b] or
            [[a1, b1], [a2, b2], ...].
        img_source (np.ndarray, optional): Colour source image. Random noise if None.
        apply_warp (bool): Randomly warp the interpolation maps.
        perspective (bool): Randomly apply a perspective shift to textures.
        sample_source (bool): Load a source image from ``path`` at construction.
    """

    def __init__(self, width=1000, use_natural_images=True, path="",
                 types=None, type_weights=None,
                 slope_range=None,
                 img_source=None,
                 apply_warp=True, perspective=False, sample_source=True):
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
        self.apply_perspective = perspective
        self.texture_types = types
        self.type_weights = type_weights
        self.slope_range = slope_range

        self.img_source = img_source
        self.src_h = img_source.shape[0]
        self.src_w = img_source.shape[1]
        self.files = []
        self.resulting_image = np.ones((width, width, 3), dtype=np.uint8)

        if self.use_natural_images and sample_source:
            self.source_image_sampling()

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
        row = np.random.randint(0, max(1, self.src_h - 101))
        col = np.random.randint(0, max(1, self.src_w - 101))
        return self.img_source[row:row + 100, col:col + 100]

    def generate_source_texture(self, texture_type, width=1000, region=None):
        """Generates a texture map from color noise or a bilevel texture mixer.

        Args:
            texture_type (str): "freq_noise", "texture_mixes", "sin" or "grid".
            width (int, optional): Size of the texture map. Defaults to 1000.
            region (tuple, optional): (r0, r1, c0, c1) part of the texture that
                is needed, the rest is left black. Defaults to None (all).
        """
        if texture_type == "freq_noise":
            slope = sample_slope(self.slope_range)
            return sample_color_noise(self.random_patch_selection(), width=width, slope=slope,
                                      region=region)

        if texture_type == "texture_mixes":
            # 90% sin, 10% grid
            single_color_1 = np.random.random() < 0.1
            single_color_2 = np.random.random() < 0.1
            mix_modes = [["sin"], ["grid"]]
            mix_mode = mix_modes[np.random.choice(2, p=[0.9, 0.1])]
        else:
            # texture_type in ["sin", "grid"]
            single_color_1 = True
            single_color_2 = True
            mix_mode = [texture_type]

        warp = np.random.random() < 0.5 if self.apply_warp else False
        # unused: threshold of a former "noise" mixing mode, still drawn so that
        # the random stream matches the original implementation
        np.random.randint(5, 50)
        return bilevel_texture_mixer(
            color_source_1=self.random_patch_selection(),
            color_source_2=self.random_patch_selection(),
            single_color_1=single_color_1, single_color_2=single_color_2,
            mixing_types=mix_mode, width=width, warp=warp,
            slope_range=self.slope_range, region=region)

    def source_image_sampling(self):
        """Selects a random source image from the path directory.
        Updates img_source, src_h and src_w.
        """
        if not self.files:
            self.files = list_images(self.path)
            if not self.files:
                raise FileNotFoundError(
                    f"No colour source images in {self.path!r}: set color.image_dir to a directory "
                    "of images, or color.use_natural_images=False for random colours.")
        ind = random.randint(0, len(self.files) - 1)
        self.img_source = skio.imread(self.files[ind])
        self.src_h = self.img_source.shape[0]
        self.src_w = self.img_source.shape[1]
        logger.debug("Sampling source image: %s", self.files[ind])

    def sample_texture_type(self):
        """Samples a texture type according to type_weights."""
        if len(self.texture_types) == 1:
            return self.texture_types[0]
        return np.random.choice(self.texture_types, p=self.type_weights)

    def generate_texture(self, width=100):
        """Generates a texture map of a randomly chosen type.

        Args:
            width (int, optional): The width of the texture map. Defaults to 100.

        Returns:
            np.ndarray: The generated texture map (width x width x 3, uint8).
        """
        texture_type = self.sample_texture_type()
        logger.debug("Generating texture of type: %s", texture_type)

        if texture_type == "gradient":
            color1 = np.uint8(self.img_source[np.random.randint(0, self.src_h),
                                              np.random.randint(0, self.src_w), :])
            color2 = np.uint8(self.img_source[np.random.randint(0, self.src_h),
                                              np.random.randint(0, self.src_w), :])
            return self.lin_gradient(color1, color2, angle=45)

        use_perspective = np.random.random() > 0.5
        if use_perspective and self.apply_perspective:
            logger.debug("Applying perspective shift")
            # generated at twice the size: the perspective crop is width x width.
            # Only the part of the texture read by the perspective crop is made.
            res = self.generate_source_texture(texture_type, width=2 * width,
                                               region=perspective_source_box(2 * width))
            return perspective_shift(res)
        return self.generate_source_texture(texture_type, width=width)

    def clear(self):
        self.resulting_image = np.ones((self.width, self.width, 3), dtype=np.uint8)


class Deadleaves(Textures):
    """Dead leaves image generator.

    Leaves (polygons, disks or rectangles with power-law distributed radii) are
    stacked front to back: each new leaf only paints the pixels not covered yet.
    An image is the composition of three such stacks (background, in-focus
    plainground, foreground), see compose_dead_leaves_depth_of_field.

    Args (in addition to those of Textures):
        rmin, rmax (int): Range of the leaf radii.
        power_law_exponent (float): Exponent of the radius distribution.
        texture_path (str): Directory of textures to load when ``fetch`` is used.
        shape_type (str): "poly", "disk", "rectangle" or "mix".
        use_texture (bool): Fill large leaves with textures (otherwise gradients).
        online_generation (bool): Generate a texture per leaf, instead of picking
            from a precomputed dictionary of n_textures textures.
        skip_hidden (bool): Do not generate leaves that end up entirely covered by
            closer leaves, and stop a stack once every pixel is covered. Hidden
            leaves leave the image unchanged, so the output distribution is the
            same; only the random stream differs. With skip_hidden=False the
            random numbers are drawn in the same order as the original
            implementation.
        independent_streams (bool): Draw the geometry (radius, type, position,
            shape) and the fill (colour, texture) of every leaf from their own
            random streams, seeded from the image's stream and the leaf's
            number. Changing the texture parameters then changes the textures
            but not the leaves' geometry or colours, which makes images
            generated with different parameters comparable.
    """

    def __init__(self, rmin=1, rmax=1000, power_law_exponent=3, width=1000,
                 use_natural_images=True, path="",
                 texture_path="", shape_type="poly",
                 types=None, type_weights=None,
                 slope_range=None, use_texture=True,
                 online_generation=False, apply_warp=True,
                 perspective=True,
                 img_source=None, skip_hidden=True, independent_streams=False):
        # the source image is sampled in compose_dead_leaves_depth_of_field()
        super().__init__(width=width, use_natural_images=use_natural_images, path=path,
                         types=types, type_weights=type_weights, slope_range=slope_range,
                         img_source=img_source, apply_warp=apply_warp,
                         perspective=perspective, sample_source=False)
        self.power_law_exponent = power_law_exponent
        self.texture_path = texture_path
        self.shape_type = shape_type
        self.use_texture = use_texture
        self.online_generation = online_generation
        self.n_textures = 10
        self.textures = []
        self.skip_hidden = skip_hidden
        self.independent_streams = independent_streams
        self._stream_base = 0
        self._n_stacks = 0
        self.clear()
        self.update_leaves_size_parameters(rmin, rmax)

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
        if EXACT_ORIGINAL:
            theoretical_n_shapes = theoretical_number_disks(
                0.002, self.rmin, self.rmax, self.width, self.power_law_exponent)
            interval = int(theoretical_n_shapes)
            if interval < 10:
                interval = 1
            if interval > 10:
                interval = interval // 10
        else:
            # background, middle ground and foreground: 10, 1 and 0.5 x interval shapes
            shapes = ["disk", "poly", "rectangle"] if self.shape_type == "mix" else [self.shape_type]
            interval = number_of_leaves(
                int(self.rmin), int(self.rmax), float(self.power_law_exponent), int(self.width),
                tuple(self.area_factors(s) for s in shapes), UNCOVERED_FRACTION)
        logger.debug("Shapes per layer step: %d", interval)
        self.interval = interval

    def area_factors(self, shape):
        """Mean area of the masks of the given shape type for the radii
        rmin, ..., rmax - 1, relative to pi * radius^2 (None for disks)."""
        if shape == "disk":
            return None
        radii, factors = SHAPE_AREA_FACTORS[shape]
        k = np.arange(self.rmin, max(self.rmax, self.rmin + 1))
        a = np.interp(np.log(k), np.log(radii), factors)
        if shape == "poly":
            a *= ((3 * k // 2) / k) ** 2
            # masks larger than 2 * rmax + 100 are scaled down (_fit_to_max_size)
            a *= np.minimum(1, (2 * self.rmax + 100.) / (2 * (3 * k // 2) + 1)) ** 2
        return tuple(a.tolist())

    def fetch_textures(self):
        """Loads n_textures textures from texture_path."""
        files = [os.path.join(self.texture_path, f)
                 for f in os.listdir(self.texture_path)]
        file_id = np.random.choice(len(files), self.n_textures)
        self.textures = [skio.imread(files[ind]) for ind in file_id]
        return self.textures

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

    def sample_radius(self):
        """Samples a shape radius from the power-law distribution."""
        radius = self.vamin + (self.vamax - self.vamin) * np.random.random()
        return int(1 / (radius ** (1. / (self.power_law_exponent - 1))))

    def sample_shape_type(self):
        shape = self.shape_type
        if shape == "mix":
            shape = random.choice(["disk", "poly", "rectangle"])
        return shape

    def _fit_to_max_size(self, shape_mask):
        if max(shape_mask.shape) >= 2 * self.rmax + 100:
            scale = (2 * self.rmax + 100.) / max(shape_mask.shape)
            new_size = (int(2 * ((shape_mask.shape[0] * scale) // 2) - 1),
                        int(2 * ((shape_mask.shape[1] * scale) // 2) - 1))
            shape_mask = np.bool_(cv2.resize(np.uint8(shape_mask), new_size,
                                             interpolation=cv2.INTER_AREA))
        return shape_mask

    def polygon_mask_size(self, radius):
        """Side length of the mask returned by generate_shape_mask for a polygon."""
        size = 2 * (3 * radius // 2) + 1
        if size >= 2 * self.rmax + 100:
            size = int(2 * ((size * ((2 * self.rmax + 100.) / size)) // 2) - 1)
        return size

    def generate_shape_mask(self, shape, radius):
        """Generates the binary mask of a shape of the given type and radius."""
        if shape == "poly":
            shape_mask = binary_polygon_generator(
                2 * (3 * radius // 2) + 1,
                n=np.random.randint(50, max(100, int(0.9 * radius))),
                allow_holes=bool(random.getrandbits(1)),
                smoothing=bool(random.getrandbits(1)))
            shape_mask = self._fit_to_max_size(shape_mask)
        elif shape == "disk":
            shape_mask = disk_mask(radius)
        elif shape == "rectangle":
            shape_mask = self._fit_to_max_size(make_rectangle_mask(radius))
        return shape_mask

    def generate_single_shape_mask(self):
        """Samples a shape radius from the power-law distribution and generates
        the corresponding binary mask.

        Returns:
            tuple: (shape_mask np.ndarray, radius int)
        """
        radius = self.sample_radius()
        return self.generate_shape_mask(self.sample_shape_type(), radius), radius

    def sample_position(self):
        """Uniform (row, col) position of a shape centre."""
        return np.random.randint(0, self.width), np.random.randint(0, self.width)

    def shape_window(self, pos, shape_h, shape_w):
        """Image window (row_min, row_max, col_min, col_max) covered by a shape
        mask of size (shape_h, shape_w) centred on pos."""
        return (max(0, pos[0] - shape_h // 2),
                min(self.width, 1 + pos[0] + shape_h // 2),
                max(0, pos[1] - shape_w // 2),
                min(self.width, 1 + pos[1] + shape_w // 2))

    def add_shape_to_binary_mask(self, shape_mask, pos=None):
        """Places a shape and marks the pixels it covers in the occlusion mask.

        Args:
            shape_mask (np.ndarray): Binary shape mask.
            pos (tuple, optional): (row, col) shape centre. Sampled uniformly if None.

        Returns:
            tuple: (row_min, row_max, col_min, col_max, visible_mask), where
            visible_mask holds the pixels of the window painted by this shape
            (those of the shape not covered by previous shapes).
        """
        shape_h, shape_w = shape_mask.shape[0], shape_mask.shape[1]
        if pos is None:
            pos = self.sample_position()
        row_min, row_max, col_min, col_max = self.shape_window(pos, shape_h, shape_w)

        shape_mask = shape_mask[
            max(0, shape_h // 2 - pos[0]):
                min(shape_h, self.width + shape_h // 2 - pos[0]),
            max(0, shape_w // 2 - pos[1]):
                min(shape_w, self.width + shape_w // 2 - pos[1])]
        uncovered = self.binary_image[row_min:row_max, col_min:col_max]
        visible_mask = uncovered & shape_mask
        uncovered &= ~visible_mask
        self.n_uncovered -= np.count_nonzero(visible_mask)
        return row_min, row_max, col_min, col_max, visible_mask

    def shape_fill(self, visible_mask, radius):
        """Samples the colour or texture filling a shape.

        Small shapes get a flat colour; larger ones a colour gradient (5%, or
        always when use_texture is False) or a texture.

        Args:
            visible_mask (np.ndarray): Visible part of the shape (its window).
            radius (int): Radius of the shape (used to decide fill type).

        Returns:
            np.ndarray: uint8 colour (3,) or texture crop (h, w, 3) to paint
            where visible_mask is True.
        """
        shape_h, shape_w = visible_mask.shape[0], visible_mask.shape[1]
        color = np.uint8(self.img_source[np.random.randint(0, self.src_h),
                                         np.random.randint(0, self.src_w), :])
        if radius < 20:
            return color

        angle = np.random.randint(0, 360)
        color_2 = np.uint8(self.img_source[np.random.randint(0, self.src_h),
                                           np.random.randint(0, self.src_w), :])
        grad_vs_texture = np.random.random() if self.use_texture else 1.0
        if grad_vs_texture > 0.95:
            k = np.random.uniform(0.1, 0.5)
            # the gradient map is (size x size); only the crop is converted to RGB
            size = 2 * max(shape_h, shape_w) + 1
            row = np.random.randint(0, max(1, size - shape_h))
            col = np.random.randint(0, max(1, size - shape_w))
            return linear_color_gradient(
                color, color_2, width=size, angle=angle, k=k, color_space="lab",
                crop=(row, col, shape_h, shape_w))

        if self.online_generation:
            texture_map = self.generate_texture(width=60 + max(shape_h, shape_w))
        else:
            texture_map = self.pick_texture(size=max(shape_h, shape_w))
        h, w = texture_map.shape[0], texture_map.shape[1]
        row = np.random.randint(0, max(1, h - shape_h))
        col = np.random.randint(0, max(1, w - shape_w))
        return texture_map[row:row + shape_h, col:col + shape_w]

    def place_next_shape(self):
        """Samples the next shape of the stack and its position.

        Returns:
            tuple: (row_min, row_max, col_min, col_max, visible_mask, radius), or
            None when skip_hidden is set and the shape is entirely hidden.
        """
        if not self.skip_hidden:
            # same order of random draws as the original implementation
            shape_mask, radius = self.generate_single_shape_mask()
            return self.add_shape_to_binary_mask(shape_mask) + (radius,)

        radius = self.sample_radius()
        shape = self.sample_shape_type()
        if shape == "poly":
            # polygons are expensive: check the bounding box before drawing them
            size = self.polygon_mask_size(radius)
            pos = self.sample_position()
            row_min, row_max, col_min, col_max = self.shape_window(pos, size, size)
            if not self.binary_image[row_min:row_max, col_min:col_max].any():
                return None
            shape_mask = self.generate_shape_mask(shape, radius)
        else:
            shape_mask = self.generate_shape_mask(shape, radius)
            pos = self.sample_position()
        placed = self.add_shape_to_binary_mask(shape_mask, pos)
        if not placed[4].any():
            return None
        return placed + (radius,)

    # kinds of independent random streams
    _GEOMETRY, _FILL, _TEXTURE_DICT = 0, 1, 2

    def _seed_stream(self, kind, stack, index):
        """Seeds numpy's (and numba's) generator for one stream, see
        independent_streams."""
        key = _mix64(_mix64(_mix64(self._stream_base * 4 + kind) + stack) + index)
        np.random.seed(key & 0xFFFFFFFF)
        if kind != self._GEOMETRY:
            # textures also draw from numba's generator
            seed_numba(key >> 33)

    def generate_stack(self, n_shapes):
        """Draws n_shapes shapes on the current canvas, front to back.

        Args:
            n_shapes (int): Number of shapes to draw.

        Returns:
            tuple: (resulting_image uint8, covered uint8), covered being 1 where
            a shape was drawn and 0 elsewhere (3 channels).
        """
        stack = self._n_stacks
        self._n_stacks += 1
        reseed_geometry = True
        for index in range(n_shapes):
            if self.skip_hidden and self.n_uncovered == 0:
                break
            if self.independent_streams and reseed_geometry:
                # the geometry stream continues through hidden shapes (which
                # depend on the geometry only) and restarts after each fill
                self._seed_stream(self._GEOMETRY, stack, index)
                reseed_geometry = False
            placed = self.place_next_shape()
            if placed is None:
                continue
            row_min, row_max, col_min, col_max, visible_mask, radius = placed
            if self.independent_streams:
                self._seed_stream(self._FILL, stack, index)
                reseed_geometry = True
            fill = self.shape_fill(visible_mask, radius)
            np.copyto(self.resulting_image[row_min:row_max, col_min:col_max], fill,
                      where=visible_mask[..., None])
        logger.debug("Dead leaves stack created (%d shapes)", n_shapes)
        return (self.resulting_image.copy(),
                np.uint8(1 - np.repeat(self.binary_image[..., np.newaxis], 3, axis=2)).copy())

    def clear(self):
        # The canvas starts at 1, not 0, as in the original implementation:
        # pixels that no shape covers keep the value 1. Those of the background
        # stack can show through in the final image as near-black specks.
        self.resulting_image = np.ones((self.width, self.width, 3), dtype=np.uint8)
        self.binary_image = np.ones((self.width, self.width), dtype=bool)
        self.n_uncovered = self.width * self.width

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

        if self.independent_streams:
            self._stream_base = np.random.randint(2 ** 31)
            self._n_stacks = 0
            self._seed_stream(self._TEXTURE_DICT, 0, 0)
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

        # Blur background, composite with in-focus plainground. Pixels covered by
        # no layer keep the background value 1 (see clear()).
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
            # blurred in float: a uint8 0/1 mask would be rounded back to 0/1
            fore_mask = cv2.GaussianBlur(fore_mask.astype(np.float32), (11, 11),
                                         sigmaX=blur_val, borderType=cv2.BORDER_DEFAULT)
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
