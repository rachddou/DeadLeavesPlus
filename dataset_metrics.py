"""Measure dataset similarity between real (GT) and synthesized images.

Usage:
  python dataset_metrics.py --real_dir real/ --fake_dir fake/
"""

import argparse
import os
import sys

from utils.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute FID, IS, KID, and Precision & Recall via torch-fidelity."
    )
    parser.add_argument("--real_dir", required=True, help="Directory with real / GT images.")
    parser.add_argument("--fake_dir", required=True, help="Directory with synthesized images.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for feature extraction (default: 64).")
    parser.add_argument("--real_crop", type=int, default=None,
                        help="Random crop size applied to real images before feature extraction.")
    parser.add_argument("--fake_crop", type=int, default=None,
                        help="Random crop size applied to fake images before feature extraction.")
    parser.add_argument("--k", type=int, default=3,
                        help="Neighbourhood size for Precision & Recall (default: 3).")
    parser.add_argument("--output", default=None,
                        help="Path to save scores as text (default: <fake_dir>/metrics.txt).")
    return parser.parse_args()


def main():
    args = parse_args()

    for name, path in [("real_dir", args.real_dir), ("fake_dir", args.fake_dir)]:
        if not os.path.isdir(path):
            sys.exit(f"[ERROR] --{name} not found: {path}")

    print(f"Real dir : {args.real_dir}")
    print(f"Fake dir : {args.fake_dir}\n")

    m = compute_metrics(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        real_crop=args.real_crop,
        fake_crop=args.fake_crop,
        prc_neighborhood=args.k,
        batch_size=args.batch_size,
    )

    lines = [
        "=" * 50,
        "Summary",
        "-" * 50,
        f"  Real dir        : {args.real_dir}",
        f"  Fake dir        : {args.fake_dir}",
        "-" * 50,
        f"  FID             : {m['frechet_inception_distance']:.4f}",
        f"  Inception Score : {m['inception_score_mean']:.4f} ± {m['inception_score_std']:.4f}",
        f"  KID             : {m['kernel_inception_distance_mean']:.6f} ± {m['kernel_inception_distance_std']:.6f}",
        f"  Precision (k={args.k}) : {m['precision']:.4f}",
        f"  Recall    (k={args.k}) : {m['recall']:.4f}",
        f"  MIND            : {m['monge_inception_distance']:.4f}",
        "=" * 50,
    ]

    output_path = args.output or os.path.join(args.fake_dir, "metrics.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nScores saved to {output_path}")


if __name__ == "__main__":
    main()
