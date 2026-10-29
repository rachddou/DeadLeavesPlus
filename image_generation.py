import logging
import os
from time import time, clock_gettime, CLOCK_MONOTONIC

import hydra
import numpy as np
import skimage.io as skio
from skimage.color import rgb2gray

from dead_leaves_generation.dead_leaves_class import Textures, Deadleaves
from dead_leaves_generation.utils.utils import set_seed

logger = logging.getLogger(__name__)


def build_generator(config, independent_streams=False):
    """Instantiates the image generator described by the config.

    Args:
        config: config object (see config/default.yaml for all keys).
        independent_streams (bool): Separate random streams for the geometry
            and the fill of each leaf (see Deadleaves), slower but makes images
            generated with different parameters comparable.

    Returns:
        Deadleaves or Textures generator.
    """
    if config.image_type == "dead_leaves":
        return Deadleaves(
            rmin=config.shape.radius_min,
            rmax=config.shape.radius_max,
            power_law_exponent=config.shape.power_law_exponent,
            width=config.image_size,
            use_natural_images=config.color.use_natural_images,
            path=config.color.image_dir,
            shape_type=config.shape.shape_type,
            types=config.texture.types,
            type_weights=config.texture.type_weights,
            slope_range=config.texture.slope_range,
            use_texture=config.texture.enabled,
            online_generation=config.texture.online_generation,
            apply_warp=config.texture.apply_warp,
            perspective=config.texture.apply_perspective,
            texture_path=config.texture.texture_path,
            independent_streams=independent_streams,
        )
    if config.image_type == "textures":
        return Textures(
            width=config.image_size,
            use_natural_images=config.color.use_natural_images,
            path=config.color.image_dir,
            types=config.texture.types,
            type_weights=config.texture.type_weights,
            slope_range=config.texture.slope_range,
            apply_warp=config.texture.apply_warp,
        )
    raise ValueError(
        f"Unknown image_type {config.image_type!r}. "
        "Expected 'dead_leaves' or 'textures'.")


def generate_image(generator, config):
    """Generates one image with a generator returned by build_generator.

    Args:
        generator: Deadleaves or Textures generator.
        config: config object (see config/default.yaml for all keys).

    Returns:
        np.ndarray: uint8 image, (H, W, 3), or (H, W) if config.color.grey.
    """
    if config.image_type == "dead_leaves":
        if config.shape.multiple_shapes:
            generator.shape_type = np.random.choice(["poly", "mix"])

        # drawn even without depth of field, so that toggling it does not
        # shift the random stream
        blur_draw = np.random.power(0.5)
        blur_val = 0.0
        if config.post_process.depth_of_field:
            scale = 5.0 if config.post_process.blur_type == "gaussian" else 10.0
            blur_val = scale * blur_draw

        generator.update_leaves_size_parameters(
            config.shape.radius_min, config.shape.radius_max)

        if config.test:
            generator.source_image_sampling()
            generator.generate_stack(15000)
        else:
            fetch = config.texture.texture_path != ""
            generator.compose_dead_leaves_depth_of_field(
                blur_type=config.post_process.blur_type,
                blur_val=blur_val,
                fetch=fetch,
            )
        generator.postprocess(
            blur=config.post_process.extra_blur,
            downscale=config.post_process.downscale,
        )

    elif config.image_type == "textures":
        generator.source_image_sampling()
        generator.resulting_image = generator.generate_texture(width=config.image_size)

    if config.color.grey:
        image = np.uint8(np.clip(255 * rgb2gray(generator.resulting_image), 0, 255))
    else:
        image = np.uint8(generator.resulting_image)
    generator.clear()
    return image


@hydra.main(version_base=None, config_path="./config/", config_name="default")
def main(config):
    """Generates dead-leaves or texture images based on the Hydra config.

    Creates the output directory, instantiates the generator, runs the generation
    loop, and saves each image to disk. When config.seed is set, image k of job
    config.task uses the seed seed + (task - 1) * n_images + k, so that the
    images are reproducible and the jobs of a --multirun do not repeat them.

    Args:
        config: Hydra config object (see config/default.yaml for all keys).
    """
    out_dir = os.path.join(config.io.path_origin, config.io.path)
    os.makedirs(out_dir, exist_ok=True)
    generator = build_generator(config)
    seed = config.get("seed")
    first_index = (config.task - 1) * config.n_images

    for k in range(config.n_images):
        t0 = time()
        if seed is not None:
            index = first_index + k
            set_seed(seed + index)
            image = generate_image(generator, config)
            filename = os.path.join(out_dir, f"im_{index:04d}.png")
        else:
            image = generate_image(generator, config)
            filename = os.path.join(out_dir, f"im_{int(100 * clock_gettime(CLOCK_MONOTONIC))}.png")
        skio.imsave(filename, image)
        logger.info("Image saved in %.2fs → %s", time() - t0, filename)


if __name__ == "__main__":
    main()
