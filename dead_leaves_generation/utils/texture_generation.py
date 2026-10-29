import numpy as np
from dead_leaves_generation.utils.interpolation_maps import sample_interpolation_map
from dead_leaves_generation.utils.colored_noise import sample_color_noise
from dead_leaves_generation.utils.kernels import mix_lab_to_rgb_u8, rgb_to_lab


def sample_slope(slope_range):
    """Samples a slope value uniformly from a union of disjoint intervals.

    Args:
        slope_range: Either [a, b] for a single interval, or [[a1,b1], [a2,b2], ...]
            for multiple disjoint intervals. Sampling probability is proportional to
            each interval's length.

    Returns:
        float: The sampled slope value.
    """
    if not isinstance(slope_range[0], (int, float)):
        # Multiple intervals: [[a1,b1], [a2,b2], ...]
        intervals = slope_range
        lengths = [interval[1] - interval[0] for interval in intervals]
        total_length = sum(lengths)
        interval_probs = [length / total_length for length in lengths]
        chosen_interval = np.random.choice(len(intervals), p=interval_probs)
        interval = intervals[chosen_interval]
        return np.random.uniform(interval[0], interval[1])
    else:
        # Single interval: [a, b]
        return np.random.uniform(slope_range[0], slope_range[1])


def bilevel_texture_mixer(color_source_1=None, color_source_2=None,
                          single_color_1=True, single_color_2=True,
                          mixing_types=None, width=1000,
                          warp=True, slope_range=None, region=None):
    """Mixes two color/texture sources with a sinusoidal or grid pattern.

    Each source is either a single colour or a coloured noise map; they are
    interpolated in the Lab colour space according to the interpolation map.

    Args:
        color_source_1 (np.ndarray, optional): Source color image 1 (H x W x 3).
            Defaults to random noise if None.
        color_source_2 (np.ndarray, optional): Source color image 2 (H x W x 3).
            Defaults to random noise if None.
        single_color_1 (bool): Whether texture 1 is a single color (True) or a
            colored noise map (False). Defaults to True.
        single_color_2 (bool): Whether texture 2 is a single color (True) or a
            colored noise map (False). Defaults to True.
        mixing_types (list, optional): Interpolation maps to multiply; subset of
            ["sin", "grid"]. Defaults to ["sin"].
        width (int): Output texture size. Defaults to 1000.
        warp (bool): Whether to apply geometric warping to the interpolation map.
            Defaults to True.
        slope_range: Frequency slope range for colored noise generation.
            Either [a, b] or [[a1,b1], [a2,b2], ...]. Defaults to [0.5, 2.5].
        region (tuple, optional): (r0, r1, c0, c1) part of the texture that is
            needed; the rest of the returned texture is left black. The noise
            and interpolation maps are still computed on the whole grid.
            Defaults to None.
    """
    if color_source_1 is None:
        color_source_1 = np.random.randint(0, 255, (100, 100, 3))
    if color_source_2 is None:
        color_source_2 = np.random.randint(0, 255, (100, 100, 3))
    if mixing_types is None:
        mixing_types = ["sin"]
    if slope_range is None:
        slope_range = [0.5, 2.5]

    if single_color_1:
        texture_map_1 = color_source_1[
            np.random.randint(0, color_source_1.shape[0], 1),
            np.random.randint(0, color_source_1.shape[1], 1), :
        ].reshape((1, 1, 3))
    else:
        texture_map_1 = sample_color_noise(color_source_1, width, sample_slope(slope_range), region)

    if single_color_2:
        texture_map_2 = color_source_2[
            np.random.randint(0, color_source_2.shape[0], 1),
            np.random.randint(0, color_source_2.shape[1], 1), :
        ].reshape((1, 1, 3))
    else:
        texture_map_2 = sample_color_noise(color_source_2, width, sample_slope(slope_range), region)

    interpolation_map = sample_interpolation_map(
        mixing_types=mixing_types, width=width, warp=warp)

    if region is not None:
        # the colour conversions are pointwise: only the needed part is converted
        r0, r1, c0, c1 = region
        crop = lambda t: t if t.shape[:2] == (1, 1) else t[r0:r1, c0:c1]
        res_image = np.zeros((width, width, 3), dtype=np.uint8)
        res_image[r0:r1, c0:c1] = mix_lab_to_rgb_u8(
            interpolation_map[r0:r1, c0:c1], rgb_to_lab(crop(texture_map_1)), rgb_to_lab(crop(texture_map_2)))
        return res_image

    # fused rgb2lab(texture / 255.) -> lab interpolation -> uint8(255 * lab2rgb)
    t1 = rgb_to_lab(texture_map_1)
    t2 = rgb_to_lab(texture_map_2)
    return mix_lab_to_rgb_u8(interpolation_map, t1, t2)
