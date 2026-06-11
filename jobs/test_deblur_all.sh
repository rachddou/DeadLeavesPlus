#!/bin/bash
#SBATCH --job-name=test_deblur_all
#SBATCH -C v100-32g
#SBATCH -A cjz@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs_slurm/test_deblur_all_%j.out
#SBATCH --error=logs_slurm/test_deblur_all_%j.out

module purge
module load arch/v100

set -x

cd $WORK/DeadLeavesPlus

source .venv/bin/activate

BASE=/lustre/fswork/projects/rech/cjz/udz45zt
export TORCH_HOME=$BASE/models_zoo/torch_cache

KODAK24="$WORK/datasets/datasets_drunet/Kodak24"
CBSD68="$WORK/datasets/datasets_drunet/CBSD68"
KERNEL_PATH="$WORK/kernels/Levin09.mat"

DEBLUR_MODELS=(
    "deblur_DL_h100"
    "deblur_VL_h100"
    "deblur_waterlooDB_h100"
)

for MODEL_DIR in "${DEBLUR_MODELS[@]}"; do
    MODEL_PATH="${BASE}/DeadLeavesPlus/TRAINING_LOGS/${MODEL_DIR}/net.pth"

    echo "========================================"
    echo "Model: ${MODEL_DIR}"
    echo "========================================"

    python3 launcher_test.py \
        --task               deblur \
        --model              UNetRes \
        --model_path         "${MODEL_PATH}" \
        --inputs             "${KODAK24}" "${CBSD68}" \
        --kernel_path        "${KERNEL_PATH}" \
        --noise_sigma_deblur 10 \
        --kernel_indices     5 \
        --save_dir           "deblur_results/${MODEL_DIR}" \
        --suffix             "sigma10_k5" \
        --save \
        --save_blurry
done
