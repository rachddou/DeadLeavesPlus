#!/bin/bash
#SBATCH --job-name=train_deblur              # nom du job
#SBATCH -C v100-32g                          # reserver uniquement des GPU V100 32 Go
##SBATCH -C v100-16g                         # decommenter pour reserver uniquement des GPU V100 16 Go
##SBATCH --partition=gpu_p2                  # decommenter pour la partition gpu_p2 (GPU V100 32 Go)
##SBATCH -C a100                             # decommenter pour la partition gpu_p5 (GPU A100 80 Go)
#SBATCH -C h100                             # decommenter pour la partition gpu_p6 (GPU H100 80 Go)
#SBATCH -A cjz@h100
#SBATCH --nodes=1                            # on demande un noeud
#SBATCH --ntasks-per-node=1                  # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                         # nombre de GPU par noeud
##SBATCH --cpus-per-task=10                   # nombre de CPU par tache pour gpu_p2 (1/4 des CPU du noeud 4-GPU V100)
##SBATCH --cpus-per-task=8                   # nombre de CPU par tache pour gpu_p5 (A100)
#SBATCH --cpus-per-task=24                  # nombre de CPU par tache pour gpu_p6 (H100)
#SBATCH --hint=nomultithread                 # hyperthreading desactive
#SBATCH --time=19:59:00                      # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=logs_slurm/train_deblur_%j.out
#SBATCH --error=logs_slurm/train_deblur_%j.out

# Nettoyage des modules charges en interactif et herites par defaut
module purge

module load arch/h100

# Echo des commandes lancees
set -x

DATASETS=$WORK/datasets/datasets_drunet

cd $WORK/DeadLeavesPlus

source .venv/bin/activate

python3 launcher_train_deblur.py \
    --train_dir    $DATASETS/WaterlooDB \
    --val_dir      $DATASETS/Kodak24 \
    --kernel_path  $WORK/kernels/Levin09.mat \
    --model        UNetRes \
    --patch_size   128 \
    --batch_size   32 \
    --lr           1e-4 \
    --max_steps    600000 \
    --steps_per_lr 100000 \
    --grad_clip \
    --log_dir      deblur_waterlooDB_h100 \
    --log_every    500 \
    --val_every    5000 \
    --save_every   10000 \
    --num_workers  10
