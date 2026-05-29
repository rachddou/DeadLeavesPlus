#!/bin/bash
#SBATCH --job-name=train_inpaint_DL_h100
#SBATCH -C h100
#SBATCH -A cjz@h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=19:50:00
#SBATCH --output=logs_slurm/train_inpaint_%j.out
#SBATCH --error=logs_slurm/train_inpaint_%j.out

module purge
module load arch/h100

set -x

cd $WORK/DeadLeavesPlus

source .venv/bin/activate

python3 launcher_train_inpaint.py \
    --train_dir    $WORK/DeadLeavesPlus/datasets/dead_leaves_v2 \
    --val_dir      $WORK/datasets/datasets_drunet/Kodak24 \
    --model        UNetRes \
    --patch_size   128 \
    --batch_size   32 \
    --p_keep_min   0.3 \
    --p_keep_max   0.9 \
    --lr           1e-4 \
    --max_steps    600000 \
    --steps_per_lr 100000 \
    --grad_clip \
    --log_dir      inpaint_DL_h100 \
    --log_every    500 \
    --val_every    5000 \
    --save_every   10000 \
    --num_workers  10
