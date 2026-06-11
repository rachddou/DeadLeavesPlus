import argparse
import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

#!/usr/bin/env python3

import matplotlib.pyplot as plt


def list_image_files(directory: Path) -> List[Path]:
    exts = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}
    return [p for p in sorted(directory.iterdir()) if p.suffix.lower() in exts and p.is_file()]


def load_grayscale(path: Path) -> np.ndarray:
    img = Image.open(path).convert('L')
    return np.asarray(img, dtype=np.float32)


def center_square_crop(arr: np.ndarray, side: int) -> np.ndarray:
    h, w = arr.shape
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return arr[y0:y0 + side, x0:x0 + side]


def compute_power_spectrum(img: np.ndarray) -> np.ndarray:
    # Optional mean removal to reduce DC spike
    img = img - img.mean()
    
    # Apply 2D Hann window to reduce edge artifacts (cross pattern)
    h, w = img.shape
    hann_y = np.hanning(h).reshape(-1, 1)
    hann_x = np.hanning(w).reshape(1, -1)
    window = hann_y * hann_x
    img = img * window
    
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    power = (np.abs(f_shift) ** 2)
    return power


def aggregate_average_power(image_paths: List[Path]) -> Tuple[np.ndarray, int]:
    # First pass: find global minimal square side
    min_side = None
    sizes = []
    for p in image_paths:
        try:
            with Image.open(p) as im:
                im = im.convert('L')
                w, h = im.size
        except Exception:
            continue
        side = min(w, h)
        sizes.append(side)
        min_side = side if min_side is None else min(min_side, side)
    if min_side is None:
        raise RuntimeError("No valid images found.")

    acc = None
    count = 0
    for p in image_paths:
        print(count)
        try:
            arr = load_grayscale(p)
        except Exception:
            continue
        side = min(arr.shape)
        arr_sq = center_square_crop(arr, side)
        if side != min_side:
            # Center crop down further to min_side
            arr_sq = center_square_crop(arr_sq, min_side)
        power = compute_power_spectrum(arr_sq)
        if acc is None:
            acc = np.zeros_like(power, dtype=np.float64)
        acc += power
        count += 1
    if count == 0:
        raise RuntimeError("Failed to process any images.")
    avg_power = acc / count
    return avg_power, min_side


def plot_log_contours(power: np.ndarray, output: Path, levels: int = 12, cmap: str = 'magma', sigma: float = 1.5):
    log_power = np.log10(1.0 + power)
    # Apply Gaussian smoothing to regularize noisy contours
    log_power_smooth = gaussian_filter(log_power, sigma=sigma)
    plt.figure(figsize=(6,6), dpi=150)
    plt.contour(log_power_smooth, levels=levels, cmap=cmap)
    plt.title("Average Squared 2D Fourier Spectrum (log10 domain)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Average squared 2D Fourier spectrum over images.")
    parser.add_argument("--input_dir", type=str, help="Input directory containing images.")
    parser.add_argument("-o", "--output", type=str, default="avg_spectrum_contours.png",
                        help="Output contour plot filename.")
    parser.add_argument("--levels", type=int, default=20, help="Number of contour levels.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Not a directory: {input_dir}")

    paths = list_image_files(input_dir)[:1000]
    if not paths:
        raise SystemExit("No image files found in directory.")

    avg_power, side = aggregate_average_power(paths)
    plot_log_contours(avg_power, Path(args.output), levels=args.levels)

    print(f"Processed {len(paths)} images. Square side used: {side}. Saved plot to {args.output}")


if __name__ == "__main__":
    main()