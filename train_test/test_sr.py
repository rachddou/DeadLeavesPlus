"""
Test a SwinIR super-resolution model on HR datasets.
LR inputs are generated on-the-fly from modcropped HR images using KAIR's
MATLAB-equivalent bicubic downsampler (imresize_np), ensuring comparability
with published SR benchmarks.

Reports PSNR (Y channel), SSIM (Y channel), and LPIPS (RGB).

Nat.pth files use a {'params': state_dict} wrapper; all others are raw
OrderedDicts — both are handled automatically.
"""
import os
import sys
import math
from time import time

import numpy as np
import cv2
import torch
import lpips
from skimage.metrics import structural_similarity as compare_ssim

# --------------------------------------------------------------------------- #
# KAIR imports                                                                  #
# --------------------------------------------------------------------------- #
_KAIR = os.path.join(os.path.dirname(__file__), '..', 'KAIR')
sys.path.insert(0, os.path.abspath(_KAIR))
from models.network_swinir import SwinIR          # noqa: E402
from utils.utils_image import imresize_np         # noqa: E402

from utils.utils import init_logger_ipol


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _modcrop(img_hwc, scale):
    """Crop HWC image so H and W are multiples of scale."""
    h, w = img_hwc.shape[:2]
    return img_hwc[:h - h % scale, :w - w % scale, :]


def _rgb_to_y(img_rgb_f32):
    """HWC float32 [0,1] RGB → float64 Y channel (normalised to [0,1])."""
    r, g, b = img_rgb_f32[:, :, 0], img_rgb_f32[:, :, 1], img_rgb_f32[:, :, 2]
    y = 16.0 + 65.481 * r + 128.553 * g + 24.966 * b
    return y / 255.0


def _psnr_y(sr, hr):
    """PSNR on Y channel; inputs are HWC float32 [0,1] RGB."""
    sr_y = _rgb_to_y(np.clip(sr, 0.0, 1.0))
    hr_y = _rgb_to_y(np.clip(hr, 0.0, 1.0))
    mse = np.mean((sr_y - hr_y) ** 2)
    if mse < 1e-10:
        return 100.0
    return -10.0 * math.log10(mse)


def _ssim_y(sr, hr):
    """SSIM on Y channel."""
    sr_y = _rgb_to_y(np.clip(sr, 0.0, 1.0))
    hr_y = _rgb_to_y(np.clip(hr, 0.0, 1.0))
    return compare_ssim(sr_y, hr_y, data_range=1.0)


def _load_state_dict(path):
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'params' in ckpt and not any(
            k.startswith('conv_first') for k in ckpt):
        return ckpt['params']
    return ckpt


def _pad_to_multiple(img_chw, window_size):
    """Pad CHW tensor so H and W are multiples of window_size."""
    _, h, w = img_chw.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h or pad_w:
        img_chw = torch.nn.functional.pad(
            img_chw.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect'
        ).squeeze(0)
    return img_chw, h, w


# --------------------------------------------------------------------------- #
# Tester                                                                        #
# --------------------------------------------------------------------------- #

class DatasetSRTester:

    def __init__(self, args):
        torch.manual_seed(0)
        self.args = args
        log_dir = os.path.join('tests', args.save_dir)
        os.makedirs(log_dir, exist_ok=True)
        model_stem = os.path.splitext(os.path.basename(args.model_path))[0]
        log_name = f"{model_stem}_{args.suffix}" if args.suffix else model_stem
        self.logger = init_logger_ipol(os.path.join(log_dir, log_name))
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        self.model = None

    # ------------------------------------------------------------------ #

    def _load_model(self, model_path, scale):
        state_dict = _load_state_dict(model_path)
        model = SwinIR(
            upscale=scale, in_chans=3, img_size=64, window_size=8,
            img_range=1., depths=[6, 6, 6, 6], embed_dim=60,
            num_heads=[6, 6, 6, 6], mlp_ratio=2,
            upsampler='pixelshuffledirect', resi_connection='1conv',
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self.model = model.to(self.device)
        print(f'Loaded {model_path}  (lightweight SwinIR x{scale})')

    # ------------------------------------------------------------------ #

    def _list_hr_images(self, hr_dir):
        return sorted(
            os.path.join(hr_dir, f)
            for f in os.listdir(hr_dir)
            if os.path.isfile(os.path.join(hr_dir, f))
        )

    def _read_rgb(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f'Cannot read {path}')
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # ------------------------------------------------------------------ #

    def _run_model(self, lr_img):
        """HWC float32 [0,1] LR → HWC float32 [0,1] SR output."""
        t = torch.from_numpy(lr_img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        t_pad, h_orig, w_orig = _pad_to_multiple(t.squeeze(0), window_size=8)
        t_pad = t_pad.unsqueeze(0)

        scale = self.args.scale
        start = time()
        with torch.no_grad():
            sr = self.model(t_pad)
        runtime = time() - start

        sr = sr[:, :, :h_orig * scale, :w_orig * scale]
        return sr.squeeze(0).permute(1, 2, 0).cpu().numpy(), runtime

    # ------------------------------------------------------------------ #

    def _compute_metrics(self, sr, hr):
        psnr = _psnr_y(sr, hr)
        ssim = _ssim_y(sr, hr)

        def _to_lpips_tensor(img_hwc):
            t = torch.from_numpy(img_hwc.transpose(2, 0, 1)).unsqueeze(0).float()
            return (t * 2.0 - 1.0).to(self.device)

        lpips_val = self.lpips_fn(
            _to_lpips_tensor(np.clip(sr, 0, 1)),
            _to_lpips_tensor(np.clip(hr, 0, 1)),
        ).item()

        return dict(psnr=psnr, ssim=ssim, lpips=lpips_val)

    # ------------------------------------------------------------------ #

    def _log_image(self, stem, metrics, runtime):
        print(f'  [{stem}] PSNR={metrics["psnr"]:.2f} dB  '
              f'SSIM={metrics["ssim"]:.4f}  LPIPS={metrics["lpips"]:.4f}  '
              f't={runtime:.3f}s')
        self.logger.info(f'\t{stem} | PSNR={metrics["psnr"]:.4f} | '
                         f'SSIM={metrics["ssim"]:.4f} | '
                         f'LPIPS={metrics["lpips"]:.4f} | t={runtime:.4f}s')

    def _log_dataset_summary(self, name, agg):
        n = agg['count']
        msg = (f'\n--- {name} | {n} images ---\n'
               f'  PSNR  : {agg["psnr"]/n:.4f} dB\n'
               f'  SSIM  : {agg["ssim"]/n:.4f}\n'
               f'  LPIPS : {agg["lpips"]/n:.4f}')
        print(msg)
        self.logger.info(msg)

    def _log_global_summary(self, all_aggs):
        total = {k: 0.0 for k in ('psnr', 'ssim', 'lpips', 'count')}
        for agg in all_aggs.values():
            for k in total:
                total[k] += agg[k]
        n = total['count']
        msg = (f'\n{"="*60}\n'
               f'GLOBAL SUMMARY — {len(all_aggs)} datasets — {n} images\n'
               f'  PSNR  : {total["psnr"]/n:.4f} dB\n'
               f'  SSIM  : {total["ssim"]/n:.4f}\n'
               f'  LPIPS : {total["lpips"]/n:.4f}\n'
               f'{"="*60}')
        print(msg)
        self.logger.info(msg)

    # ------------------------------------------------------------------ #

    def test_datasets(self, dataset_pairs):
        """
        dataset_pairs: list of (hr_dir, lr_dir_or_None, dataset_name) tuples.

        When lr_dir is None: modcrop HR to a multiple of scale, then generate
        LR with KAIR's MATLAB-equivalent bicubic (imresize_np) — the only way
        to guarantee pixel alignment and benchmark comparability.

        When lr_dir is provided: use those LR images directly. HR is cropped to
        (lr_H * scale) × (lr_W * scale) so the two are spatially consistent.
        """
        scale = self.args.scale
        self._load_model(self.args.model_path, scale)

        all_aggs = {}
        for hr_dir, lr_dir, name in dataset_pairs:
            if lr_dir is None:
                hr_paths = self._list_hr_images(hr_dir)
                pairs = [(p, None) for p in hr_paths]
            else:
                pairs = self._list_pairs(lr_dir, hr_dir)

            print(f'\n{name}  ({len(pairs)} images)'
                  + ('' if lr_dir is None else f'  [LR from {lr_dir}]'))
            agg = dict(psnr=0.0, ssim=0.0, lpips=0.0, count=0)

            for lr_path_or_none, hr_path_or_none in pairs:
                if lr_dir is None:
                    hr_path = lr_path_or_none   # pairs are (hr_path, None)
                    stem = os.path.splitext(os.path.basename(hr_path))[0]
                    hr = _modcrop(self._read_rgb(hr_path), scale)
                    lr = imresize_np(hr, 1.0 / scale, antialiasing=True)
                else:
                    lr_path, hr_path = lr_path_or_none, hr_path_or_none
                    stem = os.path.splitext(os.path.basename(lr_path))[0]
                    lr = self._read_rgb(lr_path)
                    hr = self._read_rgb(hr_path)
                    # Crop HR to the region covered by the LR grid
                    hr = hr[:lr.shape[0] * scale, :lr.shape[1] * scale, :]

                sr, runtime = self._run_model(lr)
                metrics = self._compute_metrics(sr, hr)

                for k in ('psnr', 'ssim', 'lpips'):
                    agg[k] += metrics[k]
                agg['count'] += 1
                self._log_image(stem, metrics, runtime)

                if self.args.save:
                    out = os.path.join('tests', self.args.save_dir,
                                       f'{stem}_sr.png')
                    cv2.imwrite(out, cv2.cvtColor(
                        (np.clip(sr, 0, 1) * 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR))

            self._log_dataset_summary(name, agg)
            all_aggs[name] = agg

        if len(all_aggs) > 1:
            self._log_global_summary(all_aggs)


# --------------------------------------------------------------------------- #
# Entry point called by launcher_test.py                                        #
# --------------------------------------------------------------------------- #

def test_sr_dataset(args):
    """
    Expects args.sr_pairs: list of (hr_dir, lr_dir_or_None, name) tuples.
    Built by launcher_test.py from --sr_hr_dirs / --sr_lr_dirs / --sr_names.
    """
    DatasetSRTester(args).test_datasets(args.sr_pairs)
