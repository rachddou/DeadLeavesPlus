import os
from train_test.dataset_test import test_dataset
from train_test.test_deblurring import test_deblur_dataset
from train_test.test_inpainting import test_inpaint_dataset
from train_test.test_sr import test_sr_dataset
import torch
import argparse


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="FFDNet_Test")
    parser.add_argument('--task'            , type=str  , default='denoise',
                        choices=['denoise', 'deblur', 'inpaint', 'sr'],
                        help='Which task to evaluate: denoise (default), deblur, or inpaint')
    parser.add_argument('--add_noise'       , type=str  , default="True")
    parser.add_argument("--input"           , type=str  , default="datasets/test_sets/Kodak24/", help='path to a single input dataset (legacy)')
    parser.add_argument("--inputs"          , type=str  , nargs='+', default=None, help='list of dataset paths to evaluate')
    parser.add_argument("--noise_sigmas"    , type=float, nargs='+', default=None, help='list of noise levels (0-255 scale)')
    parser.add_argument("--suffix"          , type=str  , default="", help='suffix to add to output name')
    parser.add_argument("--noise_map"       , type=str  , default = "single", help=" noise map mode")
    parser.add_argument("--save_dir"        , type=str  , default = "dl_denoising/", help=" noise map mode") 
    parser.add_argument("--model_path","--p", type=str  , default = "TRAINING_LOGS/mix_acutance/ckpt.pth")
    parser.add_argument("--model"           , type=str  , default='DRUNet' , help="Nmae of the model to train on")
    parser.add_argument("--angle"           , type=float, default = 5.0, help="rotation angle in degrees (used with --rot_invariance)")
    parser.add_argument("--Nmulti"          , type=int  , default = 7, help=" number of different realisations of the noise")
    parser.add_argument("--noise_sigma"     , type=float, default=25, help='noise level used on test set')
    


    parser.add_argument("--save"            , action='store_true', help="save denoised and noisy images")
    parser.add_argument("--save_res"        , action='store_true', help="save residual images")
    parser.add_argument("--no_gpu"          , action='store_true', help="run model on CPU")
    parser.add_argument("--residual"        , action='store_true', help="activates residual learning")
    parser.add_argument("--multi"           , action='store_true', help="different realisations of the noise")            
    parser.add_argument("--pieapp"          , action='store_true', help="true if we want to measure the pieapp metric")
    parser.add_argument("--blind"           , action='store_true',help="activates blind denoising")
    parser.add_argument("--scale_invariance", action='store_true', help="check scale invariance in denoising nets")
    parser.add_argument("--scale_factor"    , type=float, default=4.5, help="downscale factor for pyramid_reduce (used with --scale_invariance; 0 or 1 = identity)")
    parser.add_argument("--lr_invariance"   , action='store_true', help="check fliplr invariance in denoising nets")
    parser.add_argument("--ud_invariance"   , action='store_true', help="check flipud invariance in denoising nets")
    parser.add_argument("--rot_invariance"  , action='store_true', help="check rot invariance in denoising nets")
    parser.add_argument("--grey_color"      , action='store_true', help="true if we want to denoise each canal with the grey level trained model")
    parser.add_argument("--contrast_reduction", action='store_true', help="true if you want to reduce contrast of input images")
    parser.add_argument("--blur_invariance"  , action='store_true', help="apply Gaussian blur distortion before denoising")
    parser.add_argument("--blur_sigma"       , type=float, default=2.0, help="Gaussian blur sigma (used with --blur_invariance)")
    parser.add_argument("--hue_invariance"   , action='store_true', help="apply global hue shift distortion (RGB images only)")
    parser.add_argument("--hue_shift"        , type=float, default=30.0, help="hue rotation in degrees [0, 360] (used with --hue_invariance)")
    parser.add_argument("--contrast_invariance", action='store_true', help="apply contrast scaling distortion before denoising")
    parser.add_argument("--contrast_factor"  , type=float, default=5.0, help="S-curve slope k: k→0 is identity, larger k gives steeper midtone contrast (used with --contrast_invariance)")
    parser.add_argument("--fastmri"           , action='store_true', help="include FastMRI slices as an additional test dataset")
    parser.add_argument("--fastmri_root"      , type=str, default=None, help="root directory for FastMRI data (defaults to deepinv data home)")
    parser.add_argument("--fastmri_anatomy"   , type=str, default='knee', choices=['knee', 'brain'], help="FastMRI anatomy to evaluate on")
    parser.add_argument("--fastmri_slice_index", type=str, default='middle', help="slice selection for full FastMRISliceDataset: 'all', 'middle', 'random', or an int")

    # ── Deblurring-specific arguments ─────────────────────────────────────
    parser.add_argument("--kernel_path"     , type=str  , default=None,
                        help="Path to Levin09.mat (or dir of per-kernel .mat files) for deblur evaluation")
    parser.add_argument("--noise_sigma_deblur", type=float, default=0.0,
                        help="Fixed AWGN sigma (0-255 scale) added after blur during deblur evaluation (default: 0 = no noise)")
    parser.add_argument("--kernel_indices"  , type=int  , nargs='+', default=None,
                        help="Subset of kernel indices (0-based) to evaluate; default tests all kernels")
    parser.add_argument("--save_blurry"     , action='store_true',
                        help="save blurry (degraded) input images alongside deblurred outputs")
    parser.add_argument("--gray"            , action='store_true',
                        help="Force grayscale mode for deblur/inpaint evaluation")

    # ── Super-resolution-specific arguments ───────────────────────────────
    parser.add_argument("--scale"           , type=int  , default=2,
                        choices=[2, 4],
                        help="SR upscale factor (2 or 4)")
    parser.add_argument("--sr_hr_dirs"      , type=str  , nargs='+', default=None,
                        help="List of HR directories for SR evaluation")
    parser.add_argument("--sr_lr_dirs"      , type=str  , nargs='+', default=None,
                        help="Optional list of pre-existing LR directories (one per HR dir). "
                             "If omitted, LR is generated on-the-fly from HR via MATLAB bicubic.")
    parser.add_argument("--sr_names"        , type=str  , nargs='+', default=None,
                        help="Display names for each dataset (optional)")

    # ── Inpainting-specific arguments ─────────────────────────────────────
    parser.add_argument("--p_keep_values"   , type=float, nargs='+', default=None,
                        help="List of fixed p_keep values for inpaint evaluation "
                             "(e.g. 0.1 0.2 0.5). Defaults to [0.1, 0.2, 0.3, 0.5, 0.7, 0.9].")

    argspar = parser.parse_args()
    # Normalize noise levels to [0, 1]
    argspar.noise_sigma /= 255.
    if argspar.noise_sigmas is not None:
        argspar.noise_sigmas = [s / 255. for s in argspar.noise_sigmas]

    # String to bool
    argspar.add_noise = (argspar.add_noise.lower() == 'true')

    # use CUDA?
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()
    print(torch.cuda.is_available())

    if argspar.task == 'sr':
        if not argspar.sr_hr_dirs:
            parser.error("--sr_hr_dirs is required for --task sr")
        lr_dirs = argspar.sr_lr_dirs or [None] * len(argspar.sr_hr_dirs)
        if len(lr_dirs) != len(argspar.sr_hr_dirs):
            parser.error("--sr_lr_dirs and --sr_hr_dirs must have the same number of entries")
        names = argspar.sr_names or [
            os.path.basename(d.rstrip('/')) for d in argspar.sr_hr_dirs]
        argspar.sr_pairs = list(zip(argspar.sr_hr_dirs, lr_dirs, names))
        print("\n### Testing super-resolution model ###")
        print("> Parameters:")
        for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
            print('\t{}: {}'.format(p, v))
        print('\n')
        test_sr_dataset(argspar)
    elif argspar.task == 'deblur':
        if argspar.kernel_path is None:
            parser.error("--kernel_path is required for --task deblur")
        argspar.noise_sigma = argspar.noise_sigma_deblur
        print("\n### Testing deblurring model ###")
        print("> Parameters:")
        for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
            print('\t{}: {}'.format(p, v))
        print('\n')
        test_deblur_dataset(argspar)
    elif argspar.task == 'inpaint':
        print("\n### Testing inpainting model ###")
        print("> Parameters:")
        for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
            print('\t{}: {}'.format(p, v))
        print('\n')
        test_inpaint_dataset(argspar)
    else:
        print("\n### Testing denoising model ###")
        print("> Parameters:")
        for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
            print('\t{}: {}'.format(p, v))
        print('\n')
        test_dataset(argspar)