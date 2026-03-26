import numpy as np
from skimage.color import rgb2lab, lab2rgb
from dead_leaves_generation.utils.interpolation_maps import sample_grid, sample_sinusoid, sample_interpolation_map
from dead_leaves_generation.utils.geometric_perturbation import generate_perturbation
from dead_leaves_generation.utils.colored_noise import sample_color_noise


def _sample_slope_from_ranges(slope_range):
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


def bilevelTextureMixer(color_source_1=None, color_source_2=None,
                        single_color1=True, single_color2=True,
                        mixing_types=None, width=1000, thresh_val=10,
                        warp=True, slope_range=None):
    """Mixes two color/texture sources with a sinusoidal, grid, or noise pattern.

    Args:
        color_source_1 (np.ndarray, optional): Source color image 1 (H x W x 3).
            Defaults to random noise if None.
        color_source_2 (np.ndarray, optional): Source color image 2 (H x W x 3).
            Defaults to random noise if None.
        single_color1 (bool): Whether texture 1 is a single color (True) or a
            colored noise map (False). Defaults to True.
        single_color2 (bool): Whether texture 2 is a single color (True) or a
            colored noise map (False). Defaults to True.
        mixing_types (list, optional): Interpolation methods; subset of
            ["sin", "grid", "noise"]. Defaults to ["sin"].
        width (int): Output texture size. Defaults to 1000.
        thresh_val (int): Threshold half-width for noise-based binarisation.
            Defaults to 10.
        warp (bool): Whether to apply geometric warping to the interpolation map.
            Defaults to True.
        slope_range: Frequency slope range for colored noise generation.
            Either [a, b] or [[a1,b1], [a2,b2], ...]. Defaults to [0.5, 2.5].
    """
    if color_source_1 is None:
        color_source_1 = np.random.randint(0, 255, (100, 100, 3))
    if color_source_2 is None:
        color_source_2 = np.random.randint(0, 255, (100, 100, 3))
    if mixing_types is None:
        mixing_types = ["sin"]
    if slope_range is None:
        slope_range = [0.5, 2.5]

    if single_color1:
        texture_map_1 = color_source_1[
            np.random.randint(0, color_source_1.shape[0], 1),
            np.random.randint(0, color_source_1.shape[1], 1), :
        ].reshape((1, 1, 3))
    else:
        slope1 = _sample_slope_from_ranges(slope_range)
        texture_map_1 = sample_color_noise(color_source_1, width, slope1)

    if single_color2:
        texture_map_2 = color_source_2[
            np.random.randint(0, color_source_2.shape[0], 1),
            np.random.randint(0, color_source_2.shape[1], 1), :
        ].reshape((1, 1, 3))
    else:
        slope2 = _sample_slope_from_ranges(slope_range)
        texture_map_2 = sample_color_noise(color_source_2, width, slope2)

    interpolation_map = sample_interpolation_map(
        mixing_types=mixing_types, width=width, thresh_val=thresh_val, warp=warp)

    texture_map_1 = texture_map_1 / 255.
    texture_map_2 = texture_map_2 / 255.
    t1 = rgb2lab(texture_map_1)
    t2 = rgb2lab(texture_map_2)

    r = interpolation_map * (t2[..., 0] - t1[..., 0]) + t1[..., 0]
    g = interpolation_map * (t2[..., 1] - t1[..., 1]) + t1[..., 1]
    b = interpolation_map * (t2[..., 2] - t1[..., 2]) + t1[..., 2]

    res_image = lab2rgb(np.stack([r, g, b], axis=-1))
    return np.uint8(res_image * 255)


def pattern_patch_two_colors(color_1, color_2, width=100, period=None,
                              angle=45, warp=False, type="sin"):
    """Mixes two colors with a sinusoidal or grid interpolation pattern.

    Args:
        color_1: Either an image or a single RGB color.
        color_2: Either an image or a single RGB color.
        width (int): Output image width. Defaults to 100.
        period (list, optional): Period(s) of the pattern oscillations.
            Defaults to [100].
        angle (int): Rotation angle in degrees. Defaults to 45.
        warp (bool): Whether to apply atmospheric perturbation. Defaults to False.
        type (str): Interpolation map type ("sin" or "grid"). Defaults to "sin".
            Note: grid line thickness is sampled randomly by sample_grid.
    """
    if period is None:
        period = [100]

    color_1 = color_1 / 255.
    color_2 = color_2 / 255.
    t1 = rgb2lab(color_1)
    t2 = rgb2lab(color_2)

    if type == "sin":
        pattern = sample_sinusoid(width=width, angle=angle,
                                  variable_freq=(np.random.random() > 0.5))
    elif type == "grid":
        pattern = sample_grid(width=width, period=period, angle=angle)
    else:
        pattern = np.zeros((width, width))

    if warp:
        pattern = np.clip(generate_perturbation(pattern), 0, 1)

    r = pattern * (t2[..., 0] - t1[..., 0]) + t1[..., 0]
    g = pattern * (t2[..., 1] - t1[..., 1]) + t1[..., 1]
    b = pattern * (t2[..., 2] - t1[..., 2]) + t1[..., 2]

    res_image = lab2rgb(np.stack([r, g, b], axis=-1))
    return np.uint8(res_image * 255)
