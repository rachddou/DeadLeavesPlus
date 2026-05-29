#!/bin/bash
# Cross-domain adaptation benchmark: UCMLD (aerial) + FastMRI (medical)
# Tests all DRUNet variants at noise levels 25 and 50.

BASE=/lustre/fswork/projects/rech/cjz/udz45zt
DATASETS=$BASE/datasets/datasets_drunet
MODELS=$BASE/DeadLeavesPlus/models_zoo
FASTMRI_ROOT=$BASE/datasets/singlecoil_test
export TORCH_HOME=$BASE/models_zoo/torch_cache
d
INPUTS="$DATASETS/UCMLD_testset"

for model in drunet_nat drunet_dl drunet_DLTextures drunet_CLEVR drunet_vl drunet_FractalDB drunet_GTAV; do
    python3 launcher_test.py \
        --inputs $INPUTS \
        --noise_sigmas 25 50 \
        --p $MODELS/$model.pth \
        --fastmri \
        --fastmri_root $FASTMRI_ROOT \
        --fastmri_anatomy knee \
        --suffix cross_domain
done
