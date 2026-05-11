#!/bin/bash

BASE=/lustre/fswork/projects/rech/cjz/udz45zt
DATASETS=$BASE/datasets/datasets_drunet
MODELS=$BASE/DeadLeavesPlus/models_zoo

INPUTS="$DATASETS/CBSD68 $DATASETS/Kodak24 $DATASETS/Urban100 $DATASETS/mcmaster/ $DATASETS/bokeh/bokeh/"

for model in drunet_GTAV drunet_DLTextures drunet_CLEVR drunet_vl drunet_dl drunet_FractalDB; do
    python3 launcher_test.py --inputs $INPUTS --noise_sigmas 25 50 --p $MODELS/$model.pth
done