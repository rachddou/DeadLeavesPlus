from train_test.train_inpaint import train_inpaint
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Random-mask Inpainting — deepinv physics + Bernoulli masks"
    )

    # Architecture
    parser.add_argument("--model", type=str, default="UNetRes",
                        choices=["UNetDeblur", "UNetRes"],
                        help="UNetDeblur: plain Conv-BN-ReLU UNet (faster); "
                             "UNetRes: UNet with residual blocks (default: UNetRes)")

    # Data
    parser.add_argument("--train_dir",  type=str, required=True,
                        help="Folder of training images")
    parser.add_argument("--val_dir",    type=str, required=True,
                        help="Folder of validation images")
    parser.add_argument("--gray",       action='store_true',
                        help="Grayscale mode (default: RGB)")
    parser.add_argument("--patch_size", type=int, default=128,
                        help="Training patch size (default: 128)")

    # Mask — Bernoulli split ratio
    parser.add_argument("--p_keep_min", type=float, default=0.3,
                        help="Min fraction of pixels observed (default: 0.3)")
    parser.add_argument("--p_keep_max", type=float, default=0.9,
                        help="Max fraction of pixels observed (default: 0.9)")

    # TV proxy
    parser.add_argument("--use_proxy",      action='store_true',
                        help="Use TV-Chambolle PGD proxy instead of plain zero-fill "
                             "(slower per step, but faster convergence — especially "
                             "at low p_keep)")
    parser.add_argument("--proxy_pgd_iters", type=int,   default=5,
                        help="Outer PGD iterations for TV proxy (default: 5)")
    parser.add_argument("--proxy_tv_iters",  type=int,   default=15,
                        help="Inner Chambolle TV iterations per PGD step (default: 15)")
    parser.add_argument("--proxy_lam",       type=float, default=0.05,
                        help="TV regularisation weight for proxy (default: 0.05)")

    # Optimisation
    parser.add_argument("--batch_size",   "--bs",  type=int,   default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--lr",                    type=float, default=1e-4,
                        help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--min_lr",                type=float, default=1e-7,
                        help="Learning rate floor (default: 1e-7)")
    parser.add_argument("--max_steps",             type=int,   default=600_000,
                        help="Total gradient steps (default: 600 000)")
    parser.add_argument("--steps_per_lr",          type=int,   default=100_000,
                        help="Steps between LR halvings (default: 100 000)")
    parser.add_argument("--grad_clip",             action='store_true',
                        help="Clip gradients to norm 0.5")

    # Logging / saving
    parser.add_argument("--log_dir",    type=str, default="inpaint_bernoulli",
                        help="Sub-directory under TRAINING_LOGS/ (default: inpaint_bernoulli)")
    parser.add_argument("--log_every",  type=int, default=500,
                        help="Steps between train-log entries (default: 500)")
    parser.add_argument("--val_every",  type=int, default=5000,
                        help="Steps between validation runs (default: 5000)")
    parser.add_argument("--save_every", type=int, default=10_000,
                        help="Steps between checkpoint saves (default: 10 000)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader worker processes (default: 8)")

    # Wandb
    parser.add_argument("--wandb",         action='store_true',
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="inpainting",
                        help="W&B project name (default: inpainting)")
    parser.add_argument("--wandb_run",     type=str, default=None,
                        help="W&B run name (defaults to --log_dir)")

    # Resume
    parser.add_argument("--resume_training", action='store_true',
                        help="Resume from --check_point")
    parser.add_argument("--check_point", "--cp", type=str, default="ckpt.pth",
                        help="Checkpoint filename to resume from (default: ckpt.pth)")

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    print("\n### Random-mask Inpainting — deepinv physics + Bernoulli masks ###")
    print("> Parameters:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")
    print()

    train_inpaint(args)
