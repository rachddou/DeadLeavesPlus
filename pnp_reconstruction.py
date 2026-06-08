"""
DPIR image reconstruction using a DRUNet denoiser and DeepInverse.

Implements the DPIR algorithm (Zhang et al., 2021) via Half-Quadratic Splitting
(HQS) with a log-spaced decreasing noise schedule from 49/255 down to the
measurement noise level over 8 iterations.

Supports two inverse problems:
  - inpainting  : random Bernoulli mask
  - deblurring  : convolution with a Levin09 kernel + Gaussian noise

Example usage
-------------
# Inpainting on Set14
python pnp_reconstruction.py \
    --task inpainting \
    --inputs Set14/image_SRF_2/HR \
    --model_path models_zoo/drunet_dl.pth \
    --mask 0.3 \
    --save_dir pnp_inpainting/drunet_dl

# Deblurring on Set14
python pnp_reconstruction.py \
    --task deblurring \
    --inputs Set14/image_SRF_2/HR \
    --model_path models_zoo/drunet_dl.pth \
    --kernel_path datasets/Levin09.mat \
    --noise_sigma 2.55 \
    --save_dir pnp_deblurring/drunet_dl
"""

import os
import argparse
import numpy as np
import torch
import skimage.io as skio
import lpips as lpips_lib
import deepinv as dinv
from deepinv.optim import HQS
from deepinv.optim.prior import PnP
from deepinv.optim.data_fidelity import L2
from deepinv.optim.dpir import get_DPIR_params

from utils.utils import init_logger_ipol


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _load_kernels(kernel_path):
    from train_test.train_deblur import load_levin_kernels
    return load_levin_kernels(kernel_path)


def _make_inpainting_physics(img_size, mask, device):
    return dinv.physics.Inpainting(img_size=img_size, mask=mask, device=device)


def _make_deblurring_physics(img_size, kernel_np, noise_sigma, device):
    f = torch.from_numpy(kernel_np).unsqueeze(0).unsqueeze(0).float()
    return dinv.physics.BlurFFT(
        img_size=img_size,
        filter=f,
        device=device,
        noise_model=dinv.physics.GaussianNoise(sigma=noise_sigma / 255.0),
    )


def _tv_warm_start(y, physics, device, max_iter=20, lambd=0.05):
    """TV-PGD initialisation for inpainting (no closed-form A^+ available)."""
    data_fidelity = L2()
    prior_tv = dinv.optim.TVPrior()
    norm_A2 = physics.compute_sqnorm(y, tol=1e-4, verbose=False).item()
    stepsize = 1.9 / norm_A2
    x_k = torch.zeros_like(y, device=device)
    with torch.no_grad():
        for _ in range(max_iter):
            u = x_k - stepsize * data_fidelity.grad(x_k, y, physics)
            x_k = prior_tv.prox(u, gamma=lambd * stepsize)
    return x_k


def _build_dpir_model(noise_level_img, denoiser, device):
    """Instantiate the HQS/DPIR model for a given noise level.

    noise_level_img: float in [0, 1] (e.g. noise_sigma / 255).
    For inpainting (no measurement noise) use a small floor value (0.01).
    """
    sigma_denoiser, stepsize, max_iter = get_DPIR_params(
        noise_level_img, device=device
    )
    prior = PnP(denoiser=denoiser)
    model = HQS(
        prior=prior,
        data_fidelity=L2(),
        stepsize=stepsize,
        sigma_denoiser=sigma_denoiser,
        early_stop=False,
        max_iter=max_iter,
        verbose=False,
    )
    model.eval()
    return model


def _load_image(path, device):
    img = skio.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)


def _list_images(directory):
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in exts
    )


def _to_lpips_tensor(t):
    return t.float() * 2.0 - 1.0


def _save(tensor_bchw, path):
    img = tensor_bchw[0].cpu().numpy().transpose(1, 2, 0)
    skio.imsave(path, (np.clip(img, 0, 1) * 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
# Evaluation loop                                                               #
# --------------------------------------------------------------------------- #

def run(args):
    os.makedirs(os.path.join('tests', args.save_dir), exist_ok=True)
    log_path = os.path.join('tests', args.save_dir, args.save_dir.replace('/', '_'))
    if args.suffix:
        log_path += f'_{args.suffix}'
    logger = init_logger_ipol(log_path)

    psnr_metric = dinv.metric.PSNR()
    lpips_fn = lpips_lib.LPIPS(net='alex').to(args.device)

    denoiser = dinv.models.DRUNet(pretrained=args.model_path, device=args.device)
    denoiser.eval()

    # Build the DPIR model — noise level drives the schedule
    if args.task == 'inpainting':
        # No measurement noise: use the DPIR floor value (0.01)
        noise_level = max(0.01, args.noise_sigma / 255.0)
    else:
        noise_level = args.noise_sigma / 255.0

    dpir = _build_dpir_model(noise_level, denoiser, args.device)

    kernels = None
    if args.task == 'deblurring':
        kernels = _load_kernels(args.kernel_path)
        if args.kernel_indices is not None:
            kernels = [kernels[i] for i in args.kernel_indices]

    all_psnr, all_lpips = [], []

    for img_dir in args.inputs:
        image_paths = _list_images(img_dir)
        logger.info(f'\n=== {img_dir} ({len(image_paths)} images) ===')
        print(f'\n{img_dir}  ({len(image_paths)} images)')

        dir_psnr, dir_lpips = [], []

        for img_path in image_paths:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            x = _load_image(img_path, args.device)
            _, C, H, W = x.shape

            if args.task == 'inpainting':
                physics = _make_inpainting_physics((C, H, W), args.mask, args.device)
                y = physics(x)
                x_init = _tv_warm_start(y, physics, args.device)

                with torch.no_grad():
                    x_hat = torch.clamp(dpir(y, physics, init=x_init), 0.0, 1.0)

                psnr = psnr_metric(x_hat, x).item()
                lp   = lpips_fn(_to_lpips_tensor(x_hat), _to_lpips_tensor(x)).item()

                msg = f'  [{stem}] PSNR={psnr:.2f} dB  LPIPS={lp:.4f}'
                print(msg); logger.info(msg)
                dir_psnr.append(psnr); dir_lpips.append(lp)

                if args.save:
                    _save(x_hat, os.path.join('tests', args.save_dir,
                                              f'{stem}_dpir.png'))

            else:  # deblurring
                for k_idx, kernel in enumerate(kernels):
                    physics = _make_deblurring_physics(
                        (C, H, W), kernel, args.noise_sigma, args.device
                    )
                    y = physics(x)

                    with torch.no_grad():
                        x_hat = torch.clamp(dpir(y, physics), 0.0, 1.0)

                    psnr = psnr_metric(x_hat, x).item()
                    lp   = lpips_fn(_to_lpips_tensor(x_hat), _to_lpips_tensor(x)).item()

                    msg = (f'  [{stem}] kernel {k_idx:02d} | '
                           f'PSNR={psnr:.2f} dB  LPIPS={lp:.4f}')
                    print(msg); logger.info(msg)
                    dir_psnr.append(psnr); dir_lpips.append(lp)

                    if args.save:
                        _save(x_hat, os.path.join('tests', args.save_dir,
                                                   f'{stem}_k{k_idx:02d}_dpir.png'))

        n = len(dir_psnr)
        summary = (f'\n--- {img_dir} | {n} results ---\n'
                   f'  PSNR  : {np.mean(dir_psnr):.4f} dB\n'
                   f'  LPIPS : {np.mean(dir_lpips):.4f}')
        print(summary); logger.info(summary)
        all_psnr.extend(dir_psnr); all_lpips.extend(dir_lpips)

    if len(args.inputs) > 1:
        global_summary = (f'\n{"="*60}\n'
                          f'GLOBAL SUMMARY — {len(all_psnr)} results\n'
                          f'  PSNR  : {np.mean(all_psnr):.4f} dB\n'
                          f'  LPIPS : {np.mean(all_lpips):.4f}\n'
                          f'{"="*60}')
        print(global_summary); logger.info(global_summary)


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DPIR image reconstruction with DRUNet (HQS + noise schedule)'
    )

    parser.add_argument('--task', type=str, default='inpainting',
                        choices=['inpainting', 'deblurring'],
                        help='Inverse problem to solve')
    parser.add_argument('--inputs', type=str, nargs='+', required=True,
                        help='One or more directories of HR/clean test images')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to DRUNet .pth weights')

    # ── Inpainting ──────────────────────────────────────────────────────────
    parser.add_argument('--mask', type=float, default=0.3,
                        help='Fraction of pixels to keep (inpainting)')

    # ── Deblurring ──────────────────────────────────────────────────────────
    parser.add_argument('--kernel_path', type=str, default=None,
                        help='Path to Levin09.mat blur kernels (deblurring)')
    parser.add_argument('--kernel_indices', type=int, nargs='+', default=None,
                        help='Subset of kernel indices to evaluate (default: all)')
    parser.add_argument('--noise_sigma', type=float, default=2.55,
                        help='Gaussian noise std (0–255 scale); also sets the '
                             'DPIR noise schedule endpoint (use 0.01×255 ≈ 2.55 '
                             'for deblurring, ignored for inpainting)')

    # ── Output ──────────────────────────────────────────────────────────────
    parser.add_argument('--save_dir', type=str, default='pnp',
                        help='Sub-directory under tests/ for logs and images')
    parser.add_argument('--suffix', type=str, default='',
                        help='Optional suffix appended to the log filename')
    parser.add_argument('--save', action='store_true',
                        help='Save reconstructed images to --save_dir')
    parser.add_argument('--no_gpu', action='store_true',
                        help='Force CPU execution')

    args = parser.parse_args()

    if args.task == 'deblurring' and args.kernel_path is None:
        parser.error('--kernel_path is required for --task deblurring')

    args.device = 'cpu' if args.no_gpu or not torch.cuda.is_available() else 'cuda'
    print(f'Device: {args.device}')
    print(f'Task  : {args.task}')

    run(args)
