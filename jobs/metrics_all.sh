#!/bin/bash
#SBATCH --job-name=metrics_all
#SBATCH -C v100-32g
#SBATCH -A cjz@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=logs_slurm/metrics_all_%j.out
#SBATCH --error=logs_slurm/metrics_all_%j.out

module purge

BASE=/lustre/fswork/projects/rech/cjz/udz45zt
# torchvision/torch-hub weight cache (offline compute nodes read from here).
# Holds inception (FID/IS/KID), alexnet (lpips) and vgg16 (torch-fidelity prc/mind).
export TORCH_HOME=$BASE/models_zoo/torch_cache

set -x

cd $WORK/DeadLeavesPlus

source .venv/bin/activate

REAL_DIR=$WORK/datasets/datasets_drunet/WaterlooDB

# Datasets to evaluate against the real (Waterloo) set, run sequentially.
# Results are written to <fake_dir>/metrics.txt by dataset_metrics.py.
FAKE_DIRS=(
    "$SCRATCH/datasets/vl_128"
    "$SCRATCH/datasets/vl_128_disks"
    "$SCRATCH/datasets/vl_128_freqnoise"
    "$SCRATCH/datasets/vl_128_nodepth"
    "$SCRATCH/datasets/vl_128_notex_nodepth"
    "$SCRATCH/datasets/vl_128_sin"
    "$SCRATCH/datasets/GTA5"
    "$SCRATCH/datasets/fractal_subset_5000"
)

for fake_dir in "${FAKE_DIRS[@]}"; do
    echo "########## metrics: $fake_dir ##########"
    if [ ! -d "$fake_dir" ]; then
        echo "[WARN] skipping missing dir: $fake_dir"
        continue
    fi

    python dataset_metrics.py \
        --real_dir "$REAL_DIR" \
        --fake_dir "$fake_dir" \
        --real_crop 128 \
        --fake_crop 128 \
        || echo "[WARN] metrics failed for $fake_dir (exit $?), continuing"
done

echo "########## all metrics done ##########"
