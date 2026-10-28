#!/bin/bash
#SBATCH --job-name=test_pnp
#SBATCH -C v100-32g
#SBATCH -A cjz@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --output=logs_slurm/test_pnp_%j.out
#SBATCH --error=logs_slurm/test_pnp_%j.out

module purge
module load arch/v100

set -x

BASE=/lustre/fswork/projects/rech/cjz/udz45zt
PROJECT=$BASE/DeadLeavesPlus
MODELS=$PROJECT/models_zoo
export TORCH_HOME=$BASE/models_zoo/torch_cache
KERNELS="$WORK/kernels/Levin09.mat"
cd $PROJECT
source .venv/bin/activate

DATASETS="Set14/image_SRF_2/HR"

# ── Inpainting ─────────────────────────────────────────────────────────────
for model in drunet_dl drunet_vl drunet_nat; do
    for mask in 0.3 0.5; do
        python3 pnp_reconstruction.py \
            --task inpainting \
            --inputs $DATASETS \
            --model_path $MODELS/${model}.pth \
            --mask $mask \
            --save_dir "pnp_inpainting/${model}/mask_${mask}" \
            --suffix "DPIR_mask_${mask}" \
            --save
    done
done

# # ── Deblurring ─────────────────────────────────────────────────────────────
# for model in drunet_dl drunet_vl drunet_nat; do
#     for kidx in 0 1 2 3 4 5 6 7; do
#         python3 pnp_reconstruction.py \
#             --task deblurring \
#             --inputs $DATASETS \
#             --model_path $MODELS/${model}.pth \
#             --kernel_path $KERNELS \
#             --kernel_indices $kidx \
#             --noise_sigma 7 \
#             --save_dir "pnp_deblurring/${model}/kernel_${kidx}" \
#             --suffix "DPIR_levin09_k${kidx}"
#     done
# done
