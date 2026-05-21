#!/bin/bash
# Test all distortion invariances on Kodak24 at sigma=25.
# Usage: bash jobs/test_invariances.sh [model_path]

DATASET="datasets/test_sets/Kodak24"
SIGMA=25
MODEL=${1:-"TRAINING_LOGS/mix_acutance/ckpt.pth"}
SAVE_DIR="invariance_tests"

# ------------------------------------------------------------------ #
# Scale invariance  — downscale factors: linspace(0, 5, 5)           #
# factor=0 → identity (no pyramid_reduce)                            #
# ------------------------------------------------------------------ #
echo "=== Scale invariance ==="
for sf in 0 1.25 2.5 3.75 5.0; do
    python3 launcher_test.py \
        --input   "$DATASET"  \
        --noise_sigma $SIGMA  \
        --p       "$MODEL"    \
        --save_dir "$SAVE_DIR" \
        --scale_invariance    \
        --scale_factor $sf    \
        --suffix  "scale_${sf}"
done

# ------------------------------------------------------------------ #
# Rotation invariance — angles: linspace(0, 90, 5) degrees           #
# ------------------------------------------------------------------ #
echo "=== Rotation invariance ==="
for angle in 0 22.5 45 67.5 90; do
    python3 launcher_test.py \
        --input   "$DATASET"  \
        --noise_sigma $SIGMA  \
        --p       "$MODEL"    \
        --save_dir "$SAVE_DIR" \
        --rot_invariance      \
        --angle   $angle      \
        --suffix  "rot_${angle}"
done

# ------------------------------------------------------------------ #
# Contrast invariance (S-curve slope) — linspace(0, 10, 5)           #
# k=0 → identity (sigmoid approaches linear)                         #
# ------------------------------------------------------------------ #
echo "=== Contrast invariance ==="
for cf in 0 2.5 5.0 7.5 10.0; do
    python3 launcher_test.py \
        --input   "$DATASET"  \
        --noise_sigma $SIGMA  \
        --p       "$MODEL"    \
        --save_dir "$SAVE_DIR" \
        --contrast_invariance \
        --contrast_factor $cf \
        --suffix  "contrast_${cf}"
done

# ------------------------------------------------------------------ #
# Hue invariance — shift angles: linspace(0, 180, 5) degrees         #
# ------------------------------------------------------------------ #
echo "=== Hue invariance ==="
for hs in 0 45 90 135 180; do
    python3 launcher_test.py \
        --input   "$DATASET"  \
        --noise_sigma $SIGMA  \
        --p       "$MODEL"    \
        --save_dir "$SAVE_DIR" \
        --hue_invariance      \
        --hue_shift $hs       \
        --suffix  "hue_${hs}"
done

# ------------------------------------------------------------------ #
# Blur invariance — Gaussian sigma: linspace(0.5, 5, 5)              #
# ------------------------------------------------------------------ #
echo "=== Blur invariance ==="
for bs in 0.5 1.625 2.75 3.875 5.0; do
    python3 launcher_test.py \
        --input   "$DATASET"  \
        --noise_sigma $SIGMA  \
        --p       "$MODEL"    \
        --save_dir "$SAVE_DIR" \
        --blur_invariance     \
        --blur_sigma $bs      \
        --suffix  "blur_${bs}"
done
