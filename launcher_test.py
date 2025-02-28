from train_test.dataset_test import test_ffdnet_dataset
import torch
import argparse


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="FFDNet_Test")
    parser.add_argument('--add_noise'       , type=str  , default="True")
    parser.add_argument("--input"           , type=str  , default="datasets/test_sets/Kodak24/",help='path to input dataset')
    parser.add_argument("--suffix"          , type=str  , default="", help='suffix to add to output name')
    parser.add_argument("--noise_map"       , type=str  , default = "single", help=" noise map mode")
    parser.add_argument("--save_dir"        , type=str  , default = "dl_denoising/", help=" noise map mode") 
    parser.add_argument("--model_path","--p", type=str  , default = "TRAINING_LOGS/mix_acutance/ckpt.pth")
    parser.add_argument("--model"           , type=str  , default='DRUNet' , help="Nmae of the model to train on")
    parser.add_argument("--angle"           , type=int  , default = 5, help=" number of different realisations of the noise")
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
    parser.add_argument("--lr_invariance"   , action='store_true', help="check fliplr invariance in denoising nets")
    parser.add_argument("--ud_invariance"   , action='store_true', help="check flipud invariance in denoising nets")
    parser.add_argument("--rot_invariance"  , action='store_true', help="check rot invariance in denoising nets")
    parser.add_argument("--grey_color"      , action='store_true', help="true if we want to denoise each canal with the grey level trained model")
    parser.add_argument("--contrast_reduction", action='store_true', help="true if you want to reduce contrast of input images")
    argspar = parser.parse_args()
    # Normalize noises ot [0, 1]
    argspar.noise_sigma /= 255.

    # String to bool
    argspar.add_noise = (argspar.add_noise.lower() == 'true')

    # use CUDA?
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()
    print(torch.cuda.is_available())
    print("\n### Testing FFDNet model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    test_ffdnet_dataset(**vars(argspar))