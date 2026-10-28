"""Compare the optimised dead leaves generator with the original implementation.

The original ``dead_leaves_generation`` package is extracted from a git revision
into a temporary directory and imported as ``dead_leaves_generation_ref``.

* Equivalence: with identical seeds and ``skip_hidden=False`` both versions draw
  the same random numbers in the same order, so the images must agree up to
  floating point rounding (a few pixels off by 1-2 grey levels).
* Speed: both versions generate the same number of images with their default
  settings (``skip_hidden=True`` for the optimised one).

usage:
    python scripts/compare_with_reference.py --image-dir datasets/test_sets/Kodak24
"""
import argparse
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PRESETS = {
    "disk": dict(rmin=2, rmax=500, shape_type="disk", multiple_shapes=False,
                 use_texture=False, blur_type="None", downscale=True),
    # lens blur: with gaussian blur the foreground mask is now blurred in float
    # (the original rounded it back to 0/1), so foreground edges differ on purpose
    "vibrant": dict(rmin=10, rmax=500, shape_type="poly", multiple_shapes=True,
                    use_texture=True, blur_type="lens", downscale=False),
}


@njit
def _seed_numba(seed):
    np.random.seed(seed)


def extract_reference(rev, dest):
    """Writes the package at ``rev`` to dest/dead_leaves_generation_ref."""
    files = subprocess.run(["git", "ls-tree", "-r", "--name-only", rev, "dead_leaves_generation"],
                           cwd=ROOT, check=True, capture_output=True, text=True).stdout.split()
    for f in files:
        if not f.endswith(".py"):
            continue
        src = subprocess.run(["git", "show", f"{rev}:{f}"], cwd=ROOT, check=True,
                             capture_output=True, text=True).stdout
        src = re.sub(r"\bdead_leaves_generation\b", "dead_leaves_generation_ref", src)
        out = dest / f.replace("dead_leaves_generation", "dead_leaves_generation_ref", 1)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(src)
    # the original implementation loads npy/dict.npy next to the package
    (dest / "npy").symlink_to(ROOT / "npy")


def generate(module, preset, image_dir, size, seed, **kwargs):
    generator = module.Deadleaves(
        rmin=preset["rmin"], rmax=preset["rmax"], power_law_exponent=3.0, width=size,
        use_natural_images=True, path=image_dir, shape_type=preset["shape_type"],
        types=["sin", "freq_noise", "texture_mixes"], type_weights=[0.16, 0.67, 0.17],
        slope_range=[[0.5, 2.5]], use_texture=preset["use_texture"], online_generation=True,
        apply_warp=True, perspective=True, **kwargs)
    # os.listdir order is filesystem dependent: fix it so both versions see the same list
    generator.files = sorted(os.path.join(image_dir, f) for f in os.listdir(image_dir)
                             if os.path.isfile(os.path.join(image_dir, f)))
    # seed after construction: only the generation draws are expected to match
    np.random.seed(seed)
    random.seed(seed)
    _seed_numba(seed)
    if preset["multiple_shapes"]:
        generator.shape_type = np.random.choice(["poly", "mix"])
    blur_val = 0.0
    if preset["blur_type"] in ("gaussian", "lens"):
        blur_val = (5.0 if preset["blur_type"] == "gaussian" else 10.0) * np.random.power(0.5)
    t0 = time.perf_counter()
    generator.compose_dead_leaves_depth_of_field(blur_type=preset["blur_type"], blur_val=blur_val)
    generator.postprocess(blur=False, downscale=preset["downscale"])
    return np.uint8(generator.resulting_image), time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--image-dir", required=True, help="colour source images")
    parser.add_argument("--ref", default="4a6f9c7",
                        help="git revision of the original code (default: last commit before the optimisation)")
    parser.add_argument("--presets", default="disk,vibrant")
    parser.add_argument("--n", type=int, default=3, help="images per preset")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    from dead_leaves_generation import dead_leaves_class as fast

    with tempfile.TemporaryDirectory() as tmp:
        extract_reference(args.ref, Path(tmp))
        sys.path.insert(0, tmp)
        from dead_leaves_generation_ref import dead_leaves_class as ref

        for name in args.presets.split(","):
            preset = PRESETS[name]
            t_ref, t_fast = [], []
            for seed in range(args.n):
                a, ta = generate(ref, preset, args.image_dir, args.size, seed)
                b, _ = generate(fast, preset, args.image_dir, args.size, seed, skip_hidden=False)
                _, tb = generate(fast, preset, args.image_dir, args.size, seed)
                diff = np.abs(a.astype(int) - b)
                print(f"[{name}] seed {seed}: pixels differing {(diff.max(-1) > 0).mean():.2%}, "
                      f"max diff {diff.max()} | original {ta:.2f}s, optimised {tb:.2f}s")
                t_ref.append(ta)
                t_fast.append(tb)
            print(f"[{name}] mean time per image: original {np.mean(t_ref):.2f}s, "
                  f"optimised {np.mean(t_fast):.2f}s ({np.mean(t_ref) / np.mean(t_fast):.1f}x)")


if __name__ == "__main__":
    main()
