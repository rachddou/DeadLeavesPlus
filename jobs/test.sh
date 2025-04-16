python launcher_test.py --p models_zoo/drunet_vl.pth --model DRUNet --noise_map single --input datasets/testsets/kodak24/ --noise_sigma 25 --save --save_dir vibrant_leaves_drunet_kodak24/

python launcher_test.py --p models_zoo/ffdnet_vl.pth --model DRUNet --noise_map constant --input datasets/testsets/kodak24/ --noise_sigma 25 --save --save_dir vibrant_leaves_ffdnet_kodak24/