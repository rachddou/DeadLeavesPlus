from utils.dataset import prepare_data
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=\
                                  "Building the training patch database")
    parser.add_argument("--gray"                        , action='store_true',\
                        help='prepare grayscale database instead of RGB')
    # Preprocessing parameters
    parser.add_argument("--patch_size", "--p"           , type=int      , default=50    , help="Patch size")
    parser.add_argument("--stride", "--s"               , type=int      , nargs='+'     , default=None,     help="Size of stride")
    parser.add_argument("--scale"                       , type=int      , default = 1   , help="Size of stride")
    parser.add_argument("--trainset_dir"                ,type=str       , nargs='+'     , default = None,   help = "list of the directories to train on" )
    parser.add_argument("--suffix"                ,type=str       ,default = "0.10.png",   help = "suffix of the files considered" )
    parser.add_argument("--valset_dir"					,type=str		, default=None  , help='path of validation set')
    parser.add_argument("--max_number_patches", "--m"   ,type=int       , default=None  , help="Maximum number of patches")
    parser.add_argument("--aug_times", "--a"            ,type=int       ,default=1      , help="How many times to perform data augmentation")
    parser.add_argument("--nested_dir"                  ,action='store_true',help="iterates through all subdirectories if True")
    parser.add_argument("--filenames", "--fn"           , type=str      , nargs = '+'   , default=['train_vl.h5','val_color.h5'],  help="How many times to perform data augmentation")
    # Dirs
    
    args = parser.parse_args()
    if args.stride is None:
        args.stride = [args.patch_size]
    if args.gray:
        if args.trainset_dir is None:
            path = '/scratch/Raphael/DeadLeavesPlus/datasets/vibrant_leaves/'
            list_dirs = [os.path.join(path,f) for f in os.listdir(path)]
            args.trainset_dir = list_dirs
        if args.nested_dir:
            list_dirs = [os.path.join(args.trainset_dir,f) for f in os.listdir(args.trainset_dir)]
            args.trainset_dir = list_dirs
            args.stride = [60 for _ in range(len(list_dirs))]
        if args.valset_dir is None:
            args.valset_dir = 'datasets/testsets/Set12/'
    else:
        if args.trainset_dir is None:
            path = '/scratch/Raphael/DeadLeavesPlus/datasets/vibrant_leaves/'
            list_dirs = [os.path.join(path,f) for f in os.listdir(path)]
            args.trainset_dir = list_dirs
            args.stride = [60 for _ in range(len(list_dirs))]
        if args.nested_dir:
            list_dirs = [os.path.join(args.trainset_dir[0],f) for f in os.listdir(args.trainset_dir[0])]
            args.trainset_dir = list_dirs
            args.stride = [128 for _ in range(len(list_dirs))]
            
        if args.valset_dir is None:
            args.valset_dir = 'datasets/test_sets/Kodak24/'
    
    print("\n### Building databases ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    print(type(args.filenames))
    print((args.filenames))
    prepare_data(args.trainset_dir,\
                    args.valset_dir,\
                    args.patch_size,\
                    args.stride,\
                    args.scale,\
                    args.max_number_patches,\
                    filenames= args.filenames,\
                    aug_times=args.aug_times,\
                    gray_mode=args.gray,\
                    suffix = args.suffix)