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

BASE=/lustre/fswork/projects/rech/cjz/udz45zt
MODELS=$BASE/DeadLeavesPlus/models_zoo
FASTMRI_ROOT=$BASE/datasets/singlecoil_test
export TORCH_HOME=$BASE/models_zoo/torch_cache
DATASETS=$BASE/datasets/datasets_drunet
INPUTS="$DATASETS/UCMLD_testset"


cd $WORK/DeadLeavesPlus
source .venv/bin/activate

for model in drunet_nat drunet_dl drunet_DLTextures drunet_CLEVR drunet_vl drunet_FractalDB drunet_GTAV; do
    python3 launcher_test.py \
        --inputs $INPUTS \
        --noise_sigmas 25 50 \
        --p $MODELS/$model.pth \
        --fastmri \
        --fastmri_root $FASTMRI_ROOT \
        --fastmri_anatomy knee \
        --suffix cross_domain \
        --save \
        --save_dir "cross_domain/${model}"
done
