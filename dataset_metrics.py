"""Measure dataset similarity between real (GT) and synthesized images.

Usage:
  # Single k
  python dataset_metrics.py --real_dir real/ --fake_dir fake/

  # Sweep k and plot
  python dataset_metrics.py --real_dir real/ --fake_dir fake/ --sweep_k --k_max 15
  python dataset_metrics.py --real_dir real/ --fake_dir fake/ --sweep_k --plot_path curve.png
"""

import argparse
import os
import sys

import torch

from utils.metrics import (
    build_inception_feature_extractor,
    compute_fid_and_is,
    extract_features,
    improved_precision_recall,
    plot_precision_recall_curve,
    precision_recall_curve,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute FID, Inception Score, and Improved Precision & Recall."
    )
    parser.add_argument("--real_dir", required=True, help="Directory with real / GT images.")
    parser.add_argument("--fake_dir", required=True, help="Directory with synthesized images.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for feature extraction (default: 64).")
    parser.add_argument("--device", default=None,
                        help="Torch device, e.g. cuda:0 (default: cuda if available, else cpu).")
    parser.add_argument("--skip_pr", action="store_true",
                        help="Skip all Precision & Recall computation.")

    # Single-k mode
    parser.add_argument("--k", type=int, default=3,
                        help="Neighbourhood size for single-k P&R (default: 3).")

    # Sweep mode
    parser.add_argument("--sweep_k", action="store_true",
                        help="Sweep k from 1 to --k_max and plot the P&R curve.")
    parser.add_argument("--k_max", type=int, default=10,
                        help="Maximum k for --sweep_k (default: 10).")
    parser.add_argument("--plot_path", default=None,
                        help="Save plot to this path instead of showing interactively.")
    return parser.parse_args()


def main():
    args = parse_args()

    for name, path in [("real_dir", args.real_dir), ("fake_dir", args.fake_dir)]:
        if not os.path.isdir(path):
            sys.exit(f"[ERROR] --{name} not found: {path}")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Device   : {device}")
    print(f"Real dir : {args.real_dir}")
    print(f"Fake dir : {args.fake_dir}\n")

    # --- FID & Inception Score -------------------------------------------
    print("=" * 50)
    print("FID and Inception Score (torch-fidelity)")
    fid, is_mean, is_std = compute_fid_and_is(args.real_dir, args.fake_dir)
    print(f"  FID             : {fid:.4f}")
    print(f"  Inception Score : {is_mean:.4f} ± {is_std:.4f}")

    # --- Precision & Recall ---------------------------------------------
    precision = recall = curve = None

    if not args.skip_pr:
        print()
        print("=" * 50)
        model = build_inception_feature_extractor(device)
        print("  Extracting real features...")
        real_feat = extract_features(args.real_dir, model, device, args.batch_size)
        print("  Extracting fake features...")
        fake_feat = extract_features(args.fake_dir, model, device, args.batch_size)
        print(f"  Real: {real_feat.shape}  |  Fake: {fake_feat.shape}\n")

        if args.sweep_k:
            print(f"Improved Precision & Recall sweep (k=1..{args.k_max})")
            curve = precision_recall_curve(real_feat, fake_feat, k_max=args.k_max)
            for k, p, r in zip(curve["k_values"], curve["precision"], curve["recall"]):
                print(f"  k={k:>2d}  Precision: {p:.4f}  Recall: {r:.4f}")
            plot_precision_recall_curve(curve, save_path=args.plot_path)
        else:
            print(f"Improved Precision & Recall (k={args.k})")
            precision, recall = improved_precision_recall(real_feat, fake_feat, k=args.k)
            print(f"  Precision (k={args.k}) : {precision:.4f}")
            print(f"  Recall    (k={args.k}) : {recall:.4f}")

    # --- Summary --------------------------------------------------------
    print()
    print("=" * 50)
    print("Summary")
    print("-" * 50)
    print(f"  FID             : {fid:.4f}")
    print(f"  Inception Score : {is_mean:.4f} ± {is_std:.4f}")
    if curve is not None:
        for k, p, r in zip(curve["k_values"], curve["precision"], curve["recall"]):
            print(f"  k={k:>2d}  Precision: {p:.4f}  Recall: {r:.4f}")
    elif precision is not None:
        print(f"  Precision (k={args.k}) : {precision:.4f}")
        print(f"  Recall    (k={args.k}) : {recall:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
