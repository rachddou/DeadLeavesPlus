import os
import glob
import random
import numpy as np
import cv2
import scipy.io
import scipy.ndimage
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from utils.models import UNetDeblur, UNetRes
from utils.utils import batch_psnr, init_logger


def _load_mat_file(path):
    """Return a dict-like object for a .mat file regardless of format version.

    scipy.io.loadmat handles v4/v5/v6; h5py handles v7.3 (HDF5-based).
    """
    try:
        return scipy.io.loadmat(path), 'scipy'
    except NotImplementedError:
        import h5py
        return h5py.File(path, 'r'), 'h5py'


def _extract_kernels_scipy(mat):
    kernels = []
    for key in ('kernels', 'kernel', 'f', 'K'):
        if key not in mat:
            continue
        raw = mat[key]
        if raw.dtype == object:  # MATLAB cell array
            for cell in raw.flat:
                k = cell.squeeze().astype(np.float32)
                if k.ndim == 2:
                    kernels.append(k / k.sum())
        else:
            k = raw.squeeze().astype(np.float32)
            if k.ndim == 2:
                kernels.append(k / k.sum())
        break
    return kernels


def _extract_kernels_h5py(h5f):
    """Handle MATLAB v7.3 (.mat) files opened with h5py.

    In v7.3, cell arrays are stored as HDF5 object references.
    """
    import h5py
    kernels = []
    for key in ('kernels', 'kernel', 'f', 'K'):
        if key not in h5f:
            continue
        obj = h5f[key]
        if isinstance(obj, h5py.Dataset) and obj.dtype == object:
            # Cell array of references — read into numpy first so .flat works
            for ref in np.array(obj).flat:
                k = np.array(h5f[ref]).squeeze().astype(np.float32)
                # h5py stores arrays transposed relative to MATLAB column-major
                if k.ndim == 2:
                    kernels.append(k.T / k.sum())
        else:
            k = np.array(obj).squeeze().astype(np.float32)
            if k.ndim == 2:
                kernels.append(k.T / k.sum())
        break
    return kernels


def load_levin_kernels(kernel_path):
    """Load kernels from a single Levin09-style .mat file or a directory of per-kernel .mat files.

    Supports both legacy scipy-readable .mat (v4/v5/v6) and MATLAB v7.3 HDF5 .mat files.
    """
    kernels = []
    if os.path.isfile(kernel_path) and kernel_path.endswith('.mat'):
        mat, backend = _load_mat_file(kernel_path)
        if backend == 'scipy':
            kernels = _extract_kernels_scipy(mat)
        else:
            kernels = _extract_kernels_h5py(mat)
            mat.close()
    elif os.path.isdir(kernel_path):
        for fpath in sorted(glob.glob(os.path.join(kernel_path, '*.mat'))):
            mat, backend = _load_mat_file(fpath)
            if backend == 'scipy':
                kernels.extend(_extract_kernels_scipy(mat))
            else:
                kernels.extend(_extract_kernels_h5py(mat))
                mat.close()
    if not kernels:
        raise ValueError(f"No valid 2-D kernels found at: {kernel_path}")
    print(f"Loaded {len(kernels)} Levin kernels")
    return kernels


def _apply_blur_hwc(img_hwc, kernel):
    out = np.empty_like(img_hwc)
    for c in range(img_hwc.shape[2]):
        out[:, :, c] = scipy.ndimage.convolve(img_hwc[:, :, c], kernel, mode='reflect')
    return out


class BlurDataset(Dataset):
    """On-the-fly dataset: random crops from image_dir blurred by random Levin kernels."""

    def __init__(self, image_dir, kernels, patch_size=128, gray=False,
                 noise_sigma_max=2.0, virtual_length=200000):
        exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
        self.image_paths = []
        for ext in exts:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")
        self.image_paths.sort()
        self.kernels = kernels
        self.patch_size = patch_size
        self.gray = gray
        self.noise_sigma_max = noise_sigma_max / 255.0
        self.virtual_length = virtual_length

    def __len__(self):
        return self.virtual_length

    def __getitem__(self, idx):
        path = self.image_paths[idx % len(self.image_paths)]
        img = cv2.imread(path)
        if img is None:
            path = random.choice(self.image_paths)
            img = cv2.imread(path)

        if self.gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[:, :, np.newaxis]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0  # H x W x C

        # Ensure image is at least patch_size in both dims
        p = self.patch_size
        h, w = img.shape[:2]
        if h < p or w < p:
            new_h = max(h, p + 4)
            new_w = max(w, p + 4)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            if img.ndim == 2:
                img = img[:, :, np.newaxis]
            h, w = img.shape[:2]

        # Random crop
        top  = np.random.randint(0, h - p + 1)
        left = np.random.randint(0, w - p + 1)
        patch = img[top:top + p, left:left + p].copy()  # p x p x C

        # Random flip / 90-degree rotation augmentation (8 symmetries)
        aug = np.random.randint(0, 8)
        patch = np.rot90(patch, k=aug % 4)
        if aug >= 4:
            patch = np.fliplr(patch)

        # Random Levin kernel
        kernel = self.kernels[np.random.randint(len(self.kernels))]

        # Blur
        blurry = _apply_blur_hwc(patch, kernel)

        # Optional tiny AWGN to model sensor noise
        if self.noise_sigma_max > 0:
            sigma = np.random.uniform(0.0, self.noise_sigma_max)
            blurry = blurry + np.random.randn(*blurry.shape).astype(np.float32) * sigma
        blurry = np.clip(blurry, 0.0, 1.0)

        clean  = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)))
        blurry = torch.from_numpy(np.ascontiguousarray(blurry.transpose(2, 0, 1)))
        return blurry, clean


def train_deblur(args):
    torch.manual_seed(42)
    np.random.seed(42)

    kernels = load_levin_kernels(args.kernel_path)
    in_ch = 1 if args.gray else 3

    dataset_train = BlurDataset(
        args.train_dir, kernels,
        patch_size=args.patch_size, gray=args.gray,
        noise_sigma_max=args.noise_sigma_max,
        virtual_length=max(500000, args.max_steps * args.batch_size),
    )
    dataset_val = BlurDataset(
        args.val_dir, kernels,
        patch_size=256, gray=args.gray,
        noise_sigma_max=args.noise_sigma_max,
        virtual_length=50,
    )
    print(f"  Train images : {len(dataset_train.image_paths)}")
    print(f"  Val   images : {len(dataset_val.image_paths)}")

    log_path = os.path.join("TRAINING_LOGS", args.log_dir)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    logger = init_logger(args)

    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed — skipping wandb logging.")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or args.log_dir,
            config=vars(args),
            mode="offline",
        )
        wandb.watch_called = False

    if args.model == 'UNetDeblur':
        net = UNetDeblur(in_nc=in_ch, out_nc=in_ch, nc=(64, 128, 256, 512))
    elif args.model == 'UNetRes':
        net = UNetRes(in_nc=in_ch, out_nc=in_ch, nc=[64, 128, 256, 512], nb=4)
    else:
        raise ValueError(f"Unknown model: {args.model}. Choose 'UNetDeblur' or 'UNetRes'.")
    model = net.cuda()
    criterion = nn.L1Loss(reduction='sum').cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loader_train = DataLoader(
        dataset_train, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    loader_val = DataLoader(
        dataset_val, batch_size=1,
        shuffle=False, num_workers=2,
    )

    step = 0
    current_lr = args.lr

    if args.resume_training:
        ckpt_file = os.path.join(log_path, args.check_point)
        if os.path.isfile(ckpt_file):
            ckpt = torch.load(ckpt_file)
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            step = ckpt['step']
            current_lr = ckpt['current_lr']
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr
            print(f"> Resumed from step {step}, lr={current_lr:.2e}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    print(f"\n> Training starts at step {step} / {args.max_steps}")
    data_iter = iter(loader_train)

    while step < args.max_steps:
        # ── LR decay every steps_per_lr steps ──────────────────────────────
        if step > 0 and step % args.steps_per_lr == 0:
            new_lr = max(current_lr * 0.5, args.min_lr)
            if new_lr != current_lr:
                current_lr = new_lr
                for pg in optimizer.param_groups:
                    pg['lr'] = current_lr
                print(f"  [step {step:7d}] LR decayed -> {current_lr:.2e}")

        # ── Fetch batch ─────────────────────────────────────────────────────
        try:
            blurry, clean = next(data_iter)
        except StopIteration:
            data_iter = iter(loader_train)
            blurry, clean = next(data_iter)

        blurry = blurry.cuda()
        clean  = clean.cuda()

        # ── Forward / backward ──────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()
        out = model(blurry)
        loss = criterion(out, clean) / (blurry.size(0) * 2)
        loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # ── Logging ─────────────────────────────────────────────────────────
        if step % args.log_every == 0:
            model.eval()
            with torch.no_grad():
                psnr = batch_psnr(torch.clamp(out.detach(), 0., 1.), clean, 1.)
            writer.add_scalar('loss',       loss.item(), step)
            writer.add_scalar('PSNR_train', psnr,        step)
            writer.add_scalar('lr',         current_lr,  step)
            if use_wandb:
                wandb.log({'loss': loss.item(), 'PSNR_train': psnr, 'lr': current_lr}, step=step)
            print(f"[step {step:7d}] loss={loss.item():.5f}  PSNR={psnr:.2f} dB  lr={current_lr:.2e}")

        # ── Validation ──────────────────────────────────────────────────────
        if step % args.val_every == 0 and step > 0:
            model.eval()
            psnr_val = 0.0
            with torch.no_grad():
                for blurry_v, clean_v in loader_val:
                    out_v = torch.clamp(model(blurry_v.cuda()), 0., 1.)
                    psnr_val += batch_psnr(out_v, clean_v.cuda(), 1.)
            psnr_val /= len(loader_val)
            writer.add_scalar('PSNR_val', psnr_val, step)
            if use_wandb:
                wandb.log({'PSNR_val': psnr_val}, step=step)
            print(f"  [val  {step:7d}] PSNR_val={psnr_val:.2f} dB")

        # ── Checkpoint ──────────────────────────────────────────────────────
        if step % args.save_every == 0 and step > 0:
            _save_checkpoint(model, optimizer, step, current_lr, args, log_path,
                             periodic=(step % (args.save_every * 10) == 0))

        step += 1

    # Final save
    _save_checkpoint(model, optimizer, step, current_lr, args, log_path,
                     periodic=True, tag='final')
    writer.close()
    if use_wandb:
        wandb.finish()
    print(f"\n> Done. Model saved to {log_path}/net.pth")


def _save_checkpoint(model, optimizer, step, current_lr, args, log_path,
                     periodic=False, tag=None):
    torch.save(model.state_dict(), os.path.join(log_path, 'net.pth'))
    ckpt = {
        'state_dict': model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'step':       step,
        'current_lr': current_lr,
        'args':       args,
    }
    torch.save(ckpt, os.path.join(log_path, 'ckpt.pth'))
    if tag:
        torch.save(ckpt, os.path.join(log_path, f'ckpt_{tag}.pth'))
    elif periodic:
        torch.save(ckpt, os.path.join(log_path, f'ckpt_step{step}.pth'))
