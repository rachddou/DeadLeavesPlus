from train_test.train import train_ffdnet,train_drunet

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FFDNet")
    parser.add_argument("--gray", action='store_true',help='train grayscale image denoising instead of RGB')
    #Training parameters
    parser.add_argument("--batch_size","--bs"   , type=int      , default=256   , help="Training batch size")
    parser.add_argument("--epochs", "--e"       , type=int      , default=80    , help="Number of total training epochs")
    parser.add_argument("--val_noiseL"          , type=float    , default=25    , help='noise level used on validation set')
    parser.add_argument("--lr"                  , type=float    , default=1e-3  , help="Initial learning rate")

    parser.add_argument("--noiseIntL"           , nargs=2       , type=int      , default=[0, 75]       , help="Noise training interval")
    parser.add_argument("--milestone"           , nargs = '+'   , type=int      , default=[50, 60,80]   , help="When to decay learning rate; should be lower than 'epochs'")
    
    parser.add_argument("--no_orthog"           , action='store_true',help="Don't perform orthogonalization as regularization")
    parser.add_argument("--resume_training"     , action='store_true',help="Don't perform orthogonalization as regularization")
    parser.add_argument("--grad_clip"           , action='store_true',help="perform gradient clipping")
    parser.add_argument("--blur"                , action='store_true',help="activates blur augmentation")
    parser.add_argument("--residual"            , action='store_true',help="activates residual learning")
    parser.add_argument("--blind"               , action='store_true',help="activates blind denoising")

    parser.add_argument("--model"               , type=str  ,               default='FFDNet'                        , help="Name of the model to train on")
    parser.add_argument("--filenames", "--fn"   , type=str  , nargs = '+',  default='datasets/h5files/train_imnat/' , help="How many times to perform data augmentation")
    # parser.add_argument("--filenames", "--fn", type=str, nargs = '+', default=['datasets/h5files/train_imnat/','datasets/h5files/train_dl/','datasets/h5files/val'], help="How many times to perform data augmentation")

    # Saving parameters
    parser.add_argument("--check_point", "--cp" , type=str, default = "ckpt.pth",help="resume training from a previous checkpoint")
    parser.add_argument("--log_dir"             , type=str, default="logs"      , help='path of log files')
    parser.add_argument("--save_every"          , type=int, default=10          ,help="Number of training steps to log psnr and perform orthogonalization")
    parser.add_argument("--save_every_epochs"   , type=int, default=50          ,help="Number of training epochs to save state")
    
    argspar = parser.parse_args()
    # Normalize noise between [0, 1]
    argspar.val_noiseL /= 255.
    argspar.noiseIntL[0] /= 255.
    argspar.noiseIntL[1] /= 255.

    print("\n### Training FFDNet model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    if argspar.model == "DRUNet":
        train_drunet(argspar)
    elif argspar.model == "FFDNet":
        train_ffdnet(argspar)
    else:
        print("Not implemented")