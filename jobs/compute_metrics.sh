#!/bin/bash
#SBATCH --job-name=dataset_metrics
#SBATCH -C h100
#SBATCH -A cjz@h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs_slurm/compute_metrics_%j.out
#SBATCH --error=logs_slurm/compute_metrics_%j.out

module purge
module load arch/h100

set -x

cd $WORK/DeadLeavesPlus

source .venv/bin/activate

REAL=$WORK/datasets/datasets_drunet/WaterlooDB
FAKE_ROOT=$SCRATCH/datasets

DATASETS=(
    vl_128
    vl_128_disks
    vl_128_nodepth
    vl_128_sin
    vl_128_freqnoise
    vl_128_notex_nodepth
)

for DSET in "${DATASETS[@]}"; do
    FAKE=$FAKE_ROOT/$DSET
    if [ ! -d "$FAKE" ]; then
        echo "Skipping $DSET — directory not found"
        continue
    fi
    echo "========================================"
    echo "Metrics for: $DSET"
    echo "========================================"
    python dataset_metrics.py \
        --real_dir "$REAL" \
        --fake_dir "$FAKE" \
        --sweep_k \
        --k_max 10 \
        --plot_path "$FAKE/pr_curve.png"
done
