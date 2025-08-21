#!/bin/bash
# comment this line if you have already dowmloaded the training dataset
python preprocessing.py --p 128 --s 100 --m 500000 --trainset_dir  "datasets/vibrantLeaves/"  --filenames "vibrantLeaves/" "val/"

python launcher_train.py --fn "datasets/h5files/vibrantLeaves/" "datasets/h5files/val/" --log_dir vibrantLeaves/ --model "DRUNet" --no_orthog --grad_clip
