#!/bin/bash

BASE=/lustre/fswork/projects/rech/cjz/udz45zt
DATASETS=$BASE/datasets/datasets_drunet
MODELS=$BASE/DeadLeavesPlus/models_zoo

export TORCH_HOME=$BASE/models_zoo/torch_cache

INPUTS="$DATASETS/CBSD68 $DATASETS/Kodak24 $DATASETS/Urban100 $DATASETS/mcmaster/ $DATASETS/bokeh/bokeh/"
#drunet_GTAV drunet_DLTextures drunet_CLEVR drunet_vl drunet_dl drunet_FractalDB
for model in ffdnet_nat; do
    python3 launcher_test.py --inputs $INPUTS --noise_sigmas 25 50 --p $MODELS/$model.pth --model FFDNet --noise_map "constant" --residual
done