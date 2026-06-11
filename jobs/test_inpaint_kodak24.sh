#!/bin/bash
#SBATCH --job-name=test_inpaint_kodak24
#SBATCH -C v100-32g
#SBATCH -A cjz@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs_slurm/test_inpaint_kodak24_%j.out
#SBATCH --error=logs_slurm/test_inpaint_kodak24_%j.out

module purge
module load arch/v100

set -x

cd $WORK/DeadLeavesPlus

source .venv/bin/activate
BASE=/lustre/fswork/projects/rech/cjz/udz45zt
export TORCH_HOME=$BASE/models_zoo/torch_cache

KODAK24="$WORK/datasets/datasets_drunet/Kodak24"
P_KEEP_VALUES="0.2 0.4 0.6"

INPAINT_MODELS=(
    "inpaint_DL_h100"
    "inpaint_VL_h100"
    "inpaint_waterlooDB_h100"
)

for MODEL_DIR in "${INPAINT_MODELS[@]}"; do
    MODEL_PATH="/lustre/fswork/projects/rech/cjz/udz45zt/DeadLeavesPlus/TRAINING_LOGS/${MODEL_DIR}/net.pth"
    SAVE_DIR="inpaint_kodak24_${MODEL_DIR}"

    echo "========================================"
    echo "Testing model: ${MODEL_PATH}"
    echo "Save dir:      tests/${SAVE_DIR}"
    echo "p_keep values: ${P_KEEP_VALUES}"
    echo "========================================"

    python3 launcher_test.py \
        --task          inpaint \
        --model         UNetRes \
        --model_path    "${MODEL_PATH}" \
        --input         "${KODAK24}" \
        --p_keep_values ${P_KEEP_VALUES} \
        --save_dir      "${SAVE_DIR}" \
        --save
done
