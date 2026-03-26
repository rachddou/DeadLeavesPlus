import logging
import os
from time import time

import hydra
import numpy as np
import skimage.io as skio
import torch
from skimage.color import rgb2gray

from dead_leaves_generation_pytorch.dead_leaves_class import Textures, Deadleaves

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./config/", config_name="default")
def main(config):
    """GPU-accelerated dead-leaves / texture image generation.

    Uses the same Hydra config as image_generation.py.  Extra top-level keys:
        batch_size (int, default 4): images generated in parallel per GPU call.
        device     (str, default ""):  empty → auto-detect CUDA/CPU.

    Args:
        config: Hydra config object (see config/default.yaml).
    """
    out_dir = os.path.join(config.io.path_origin, config.io.path)
    os.makedirs(out_dir, exist_ok=True)

    batch_size = int(getattr(config, "batch_size", 4))
    device_str = str(getattr(config, "device", ""))
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s  |  batch_size: %d", device, batch_size)

    if config.image_type == "dead_leaves":
        generator = Deadleaves(
            rmin=config.shape.radius_min,
            rmax=config.shape.radius_max,
            power_law_exponent=config.shape.power_law_exponent,
            width=config.image_size,
            use_natural_images=config.color.use_natural_images,
            path=config.color.image_dir,
            shape_type=config.shape.shape_type,
            types=list(config.texture.types),
            type_weights=list(config.texture.type_weights),
            slope_range=list(config.texture.slope_range),
            use_texture=config.texture.enabled,
            online_generation=config.texture.online_generation,
            apply_warp=config.texture.apply_warp,
            random_phase=config.texture.random_phase,
            perspective=config.texture.apply_perspective,
            texture_path=config.texture.texture_path,
            device=device,
            batch_size=batch_size,
        )
    elif config.image_type == "textures":
        generator = Textures(
            width=config.image_size,
            use_natural_images=config.color.use_natural_images,
            path=config.color.image_dir,
            types=list(config.texture.types),
            type_weights=list(config.texture.type_weights),
            slope_range=list(config.texture.slope_range),
            apply_warp=config.texture.apply_warp,
            device=device,
        )
    else:
        raise ValueError(
            f"Unknown image_type {config.image_type!r}. "
            "Expected 'dead_leaves' or 'textures'.")

    # Round up n_images to a multiple of batch_size
    n_batches = int(np.ceil(config.n_images / batch_size))
    images_remaining = config.n_images

    for batch_idx in range(n_batches):
        t0 = time()
        actual_batch = min(batch_size, images_remaining)

        if config.image_type == "dead_leaves":
            if config.shape.multiple_shapes:
                generator.shape_type = np.random.choice(["poly", "mix"])

            blur_val = 0.0
            if config.post_process.depth_of_field:
                scale = 5.0 if config.post_process.blur_type == "gaussian" else 10.0
                blur_val = scale * np.random.power(0.5)

            generator.update_leaves_size_parameters(
                config.shape.radius_min, config.shape.radius_max)
            generator.batch_size = actual_batch

            fetch = config.texture.texture_path != ""
            generator.compose_dead_leaves_depth_of_field_batch(
                blur_type=config.post_process.blur_type,
                blur_val=blur_val,
                fetch=fetch,
            )
            generator.postprocess_batch(
                blur=config.post_process.extra_blur,
                downscale=config.post_process.downscale,
            )
            images_np = generator.results_as_numpy()  # (B, H, W, 3) uint8

        elif config.image_type == "textures":
            generator.source_image_sampling()
            tex = generator.generate_texture(width=config.image_size)
            # (3, H, W) uint8 tensor → (1, H, W, 3) numpy
            images_np = tex.permute(1, 2, 0).cpu().numpy()[np.newaxis]

        # Save each image in the batch
        for i in range(actual_batch):
            img = images_np[i]
            ts = int(time() * 1e4)
            filename = os.path.join(out_dir, f"im_{ts + i:016d}.png")
            if config.color.grey:
                skio.imsave(filename,
                            np.uint8(np.clip(255 * rgb2gray(img), 0, 255)))
            else:
                skio.imsave(filename, img)
            logger.info("Image saved → %s", filename)

        generator.clear()
        images_remaining -= actual_batch
        logger.info(
            "Batch %d/%d done in %.2fs  (%d images)",
            batch_idx + 1, n_batches, time() - t0, actual_batch,
        )


if __name__ == "__main__":
    main()
