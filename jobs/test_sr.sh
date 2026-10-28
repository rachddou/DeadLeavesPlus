#!/bin/bash
#SBATCH --job-name=test_sr
#SBATCH -C v100-32g
#SBATCH -A cjz@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs_slurm/test_sr_%j.out
#SBATCH --error=logs_slurm/test_sr_%j.out

module purge
# module load arch/v100

set -x

BASE=/lustre/fswork/projects/rech/cjz/udz45zt
PROJECT=$BASE/DeadLeavesPlus
export TORCH_HOME=$BASE/models_zoo/torch_cache

cd $PROJECT
source .venv/bin/activate

# LR is generated on-the-fly from HR via MATLAB bicubic (imresize_np).
# HR directories only — no --sr_lr_dirs needed.

# ── Scale x2 ──────────────────────────────────────────────────────────────
for model in DL_E VL_E Nat; do
    python3 launcher_test.py \
        --task sr \
        --scale 2 \
        --model_path sr_models/swin_ir_x2/${model}.pth \
        --sr_hr_dirs  Set5/original  Set14/image_SRF_2/HR  DIV2K/DIV2K_valid_HR \
        --sr_names    Set5           Set14                  DIV2K_valid \
        --save_dir    "sr_x2/${model}" \
        --suffix      x2
done

# ── Scale x4 ──────────────────────────────────────────────────────────────
for model in DL_E VL_E Nat; do
    python3 launcher_test.py \
        --task sr \
        --scale 4 \
        --model_path sr_models/swin_ir_x4/${model}.pth \
        --sr_hr_dirs  Set5/original  Set14/image_SRF_4/HR  DIV2K/DIV2K_valid_HR \
        --sr_names    Set5           Set14                  DIV2K_valid \
        --save_dir    "sr_x4/${model}" \
        --suffix      x4
done
