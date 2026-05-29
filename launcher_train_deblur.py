from train_test.train_deblur import train_deblur
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blind Image Deblurring — UNet + Levin Kernels")

    # Architecture
    parser.add_argument("--model", type=str, default="UNetDeblur",
                        choices=["UNetDeblur", "UNetRes"],
                        help="UNetDeblur: plain Conv-BN-ReLU UNet (faster); UNetRes: UNet with residual blocks (default: UNetDeblur)")

    # Data
    parser.add_argument("--train_dir",    type=str, required=True,
                        help="Folder of training images")
    parser.add_argument("--val_dir",      type=str, required=True,
                        help="Folder of validation images")
    parser.add_argument("--kernel_path",  type=str, required=True,
                        help=".mat file (Levin09.mat) or directory of per-kernel .mat files")
    parser.add_argument("--gray",         action='store_true',
                        help="Grayscale mode (default: RGB)")
    parser.add_argument("--patch_size",   type=int, default=128,
                        help="Training patch size (default: 128)")
    parser.add_argument("--noise_sigma_max", type=float, default=2.0,
                        help="Max AWGN sigma (0-255 scale) added after blur to simulate sensor noise (default: 2.0)")

    # Optimisation
    parser.add_argument("--batch_size", "--bs",  type=int,   default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--lr",                  type=float, default=1e-4,
                        help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--min_lr",              type=float, default=1e-7,
                        help="Learning rate floor — training stops decaying below this (default: 1e-7)")
    parser.add_argument("--max_steps",           type=int,   default=600000,
                        help="Total gradient steps (default: 600 000)")
    parser.add_argument("--steps_per_lr",        type=int,   default=100000,
                        help="Steps between LR halvings (default: 100 000)")
    parser.add_argument("--grad_clip",           action='store_true',
                        help="Clip gradients to norm 0.5")

    # Logging / saving
    parser.add_argument("--log_dir",    type=str, default="deblur_levin",
                        help="Sub-directory under TRAINING_LOGS/ (default: deblur_levin)")
    parser.add_argument("--log_every",  type=int, default=500,
                        help="Steps between train-log entries (default: 500)")
    parser.add_argument("--val_every",  type=int, default=5000,
                        help="Steps between validation runs (default: 5000)")
    parser.add_argument("--save_every", type=int, default=10000,
                        help="Steps between checkpoint saves (default: 10 000)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader worker processes (default: 8)")

    # Wandb
    parser.add_argument("--wandb",         action='store_true',
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="deblurring",
                        help="W&B project name (default: deblurring)")
    parser.add_argument("--wandb_run",     type=str, default=None,
                        help="W&B run name (defaults to --log_dir)")

    # Resume
    parser.add_argument("--resume_training", action='store_true',
                        help="Resume from --check_point")
    parser.add_argument("--check_point", "--cp", type=str, default="ckpt.pth",
                        help="Checkpoint filename to resume from (default: ckpt.pth)")

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    print("\n### Blind Image Deblurring — UNet + Levin Kernels ###")
    print("> Parameters:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")
    print()

    train_deblur(args)
