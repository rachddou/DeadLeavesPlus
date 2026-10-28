#!/bin/bash
#SBATCH --job-name=gen_vl_128
#SBATCH --partition=cpu_p1
#SBATCH -A cjz@cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs_slurm/gen_vl_128_%j.out
#SBATCH --error=logs_slurm/gen_vl_128_%j.out

module purge
module load arch/avx512

set -x

cd $WORK/DeadLeavesPlus

source .venv/bin/activate

WORKERS=$(seq -s, 1 40)

python image_generation.py --multirun \
    task=$WORKERS \
    io.path=vl_128/ \
    hydra.launcher.n_jobs=40
