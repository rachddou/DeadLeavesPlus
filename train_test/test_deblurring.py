"""
Test a deblurring model on multiple datasets, reporting scores for every
Levin09 kernel individually and as an aggregate.
"""
import os
from time import time
import numpy as np
import cv2
import scipy.ndimage
import torch
import torch.nn as nn
import lpips

from utils.models import UNetDeblur, UNetRes
from utils.utils import (init_logger_ipol, variable_to_cv2_image,
                         remove_dataparallel_wrapper, batch_ssim, is_rgb)
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from train_test.train_deblur import load_levin_kernels

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def _apply_blur_hwc(img_hwc, kernel):
    """Convolve each channel of a H×W×C float32 image with kernel."""
    out = np.empty_like(img_hwc)
    for c in range(img_hwc.shape[2]):
        out[:, :, c] = scipy.ndimage.convolve(img_hwc[:, :, c], kernel, mode='reflect')
    return out


class DatasetDeblurTester:

    def __init__(self, args):
        torch.manual_seed(0)
        self.args = args
        log_dir = os.path.join('tests', args.save_dir)
        os.makedirs(log_dir, exist_ok=True)
        model_stem = args.model_path.split('/')[-1].split('.')[0]
        log_name = f"{model_stem}_{args.suffix}" if args.suffix else model_stem
        self.logger = init_logger_ipol(os.path.join(log_dir, log_name))
        self.dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
        self.lpips_fn = (lpips.LPIPS(net='alex').cuda()
                         if args.cuda else lpips.LPIPS(net='alex'))
        self.model = None

    # ------------------------------------------------------------------ #
    # Model loading                                                        #
    # ------------------------------------------------------------------ #

    def _load_model(self, num_channels):
        args = self.args
        if args.model == 'UNetDeblur':
            network = UNetDeblur(in_nc=num_channels, out_nc=num_channels,
                                 nc=(64, 128, 256, 512))
        elif args.model == 'UNetRes':
            network = UNetRes(in_nc=num_channels, out_nc=num_channels,
                              nc=[64, 128, 256, 512], nb=4)
        else:
            raise ValueError(f"Unknown deblur model: {args.model}. "
                             "Choose 'UNetDeblur' or 'UNetRes'.")

        model_filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                      args.model_path)
        if args.cuda:
            if 'ckpt' in model_filepath:
                state_dict = torch.load(model_filepath)['state_dict']
            else:
                state_dict = torch.load(model_filepath)
            self.model = network.cuda()
        else:
            state_dict = torch.load(model_filepath, map_location='cpu')
            state_dict = remove_dataparallel_wrapper(state_dict)
            self.model = network

        self.model.load_state_dict(state_dict)
        self.model.eval()
        print('Model loaded.\n')

    # ------------------------------------------------------------------ #
    # Data loading                                                         #
    # ------------------------------------------------------------------ #

    def _list_images(self, dataset_path):
        return sorted([
            os.path.join(dataset_path, entry)
            for entry in os.listdir(dataset_path)
            if os.path.isfile(os.path.join(dataset_path, entry))
        ])

    def _load_image(self, filepath):
        """Read image, normalize to [0, 1] float32, crop to 800×800."""
        image = cv2.imread(filepath)
        if image is None:
            raise RuntimeError(f'Could not read {filepath}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        return image[:800, :800, :]

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def _prepare_tensor(self, image):
        """Crop to multiple of 8, expand batch dim, optionally pad to even dims."""
        crop_h = 8 * (image.shape[0] // 8)
        crop_w = 8 * (image.shape[1] // 8)
        arr = image[:crop_h, :crop_w, :].transpose(2, 0, 1)
        arr = np.expand_dims(arr, 0)

        pad_h = arr.shape[2] % 2 == 1
        pad_w = arr.shape[3] % 2 == 1
        if pad_h:
            arr = np.concatenate((arr, arr[:, :, -1:, :]), axis=2)
        if pad_w:
            arr = np.concatenate((arr, arr[:, :, :, -1:]), axis=3)
        return torch.Tensor(arr.copy()), pad_h, pad_w

    def _run_model(self, clean_img, kernel):
        """Blur clean_img with kernel, add optional noise, run model.

        Returns (deblurred_tensor, clean_tensor, blurry_tensor, runtime).
        """
        args = self.args

        blurry_img = _apply_blur_hwc(clean_img, kernel)
        if args.noise_sigma > 0:
            sigma = args.noise_sigma / 255.0
            blurry_img = blurry_img + np.random.randn(*blurry_img.shape).astype(np.float32) * sigma
        blurry_img = np.clip(blurry_img, 0.0, 1.0)

        clean_tensor, pad_h, pad_w = self._prepare_tensor(clean_img)
        blurry_tensor, _, _ = self._prepare_tensor(blurry_img)

        with torch.no_grad():
            clean_t = clean_tensor.type(self.dtype)
            blurry_t = blurry_tensor.type(self.dtype)

        start_time = time()
        with torch.no_grad():
            raw_out = self.model(blurry_t)
        runtime = time() - start_time

        deblurred = torch.clamp(raw_out, 0., 1.)

        if pad_h:
            clean_t    = clean_t[:, :, :-1, :]
            deblurred  = deblurred[:, :, :-1, :]
            blurry_t   = blurry_t[:, :, :-1, :]
        if pad_w:
            clean_t    = clean_t[:, :, :, :-1]
            deblurred  = deblurred[:, :, :, :-1]
            blurry_t   = blurry_t[:, :, :, :-1]

        return deblurred, clean_t, blurry_t, runtime

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #

    def _compute_metrics(self, deblurred_tensor, clean_tensor, blurry_tensor):
        deblurred_cv2  = variable_to_cv2_image(deblurred_tensor)
        clean_cv2      = variable_to_cv2_image(clean_tensor)

        deblurred_f = np.asarray(deblurred_cv2, dtype=np.float32)
        clean_f     = np.asarray(clean_cv2, dtype=np.float32)

        mse = np.mean((deblurred_f - clean_f) ** 2)
        psnr = 10 * np.log10((255 ** 2) / (mse + 1e-8))
        psnr_blurry = compare_psnr(clean_cv2, variable_to_cv2_image(blurry_tensor))
        ssim_val = batch_ssim(deblurred_tensor, clean_tensor)

        def _to_lpips(tensor_bchw):
            return (tensor_bchw.float() * 2.0 - 1.0)

        lpips_val = self.lpips_fn(
            _to_lpips(deblurred_tensor), _to_lpips(clean_tensor)
        ).item()

        return dict(psnr=psnr, psnr_blurry=psnr_blurry, ssim=ssim_val,
                    lpips=lpips_val, deblurred_cv2=deblurred_cv2)

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #

    def _log_image(self, image_stem, kernel_idx, metrics, runtime):
        print(f"  [{image_stem}] kernel {kernel_idx:02d} | "
              f"PSNR={metrics['psnr']:.2f} dB | "
              f"LPIPS={metrics['lpips']:.4f} | t={runtime:.3f}s")
        self.logger.info(f"\t{image_stem} | kernel {kernel_idx:02d} | "
                         f"PSNR={metrics['psnr']:.4f} | "
                         f"PSNR_blurry={metrics['psnr_blurry']:.4f} | "
                         f"SSIM={metrics['ssim']:.4f} | "
                         f"LPIPS={metrics['lpips']:.4f} | "
                         f"t={runtime:.4f}s")

    def _log_kernel_summary(self, dataset_path, kernel_idx, agg):
        n = agg['count']
        msg = (f"\n--- {dataset_path} | kernel {kernel_idx:02d} | {n} images ---\n"
               f"  PSNR        : {agg['psnr']/n:.4f} dB\n"
               f"  PSNR Blurry : {agg['psnr_blurry']/n:.4f} dB\n"
               f"  SSIM        : {agg['ssim']/n:.4f}\n"
               f"  LPIPS       : {agg['lpips']/n:.4f}")
        print(msg)
        self.logger.info(msg)

    def _log_dataset_summary(self, dataset_path, all_kernel_aggs):
        total = {k: 0.0 for k in ('psnr', 'psnr_blurry', 'ssim', 'lpips', 'count')}
        for agg in all_kernel_aggs:
            for k in total:
                total[k] += agg[k]
        n = total['count']
        msg = (f"\n=== {dataset_path} | ALL {len(all_kernel_aggs)} kernels | {n} image-kernel pairs ===\n"
               f"  PSNR        : {total['psnr']/n:.4f} dB\n"
               f"  PSNR Blurry : {total['psnr_blurry']/n:.4f} dB\n"
               f"  SSIM        : {total['ssim']/n:.4f}\n"
               f"  LPIPS       : {total['lpips']/n:.4f}")
        print(msg)
        self.logger.info(msg)

    def _log_global_summary(self, all_dataset_aggs):
        """Summarize across all datasets and all kernels."""
        total = {k: 0.0 for k in ('psnr', 'psnr_blurry', 'ssim', 'lpips', 'count')}
        for aggs in all_dataset_aggs.values():
            for agg in aggs:
                for k in total:
                    total[k] += agg[k]
        n = total['count']
        msg = (f"\n{'='*60}\n"
               f"GLOBAL SUMMARY — all datasets, all kernels — {n} pairs\n"
               f"  PSNR        : {total['psnr']/n:.4f} dB\n"
               f"  PSNR Blurry : {total['psnr_blurry']/n:.4f} dB\n"
               f"  SSIM        : {total['ssim']/n:.4f}\n"
               f"  LPIPS       : {total['lpips']/n:.4f}\n"
               f"{'='*60}")
        print(msg)
        self.logger.info(msg)

    # ------------------------------------------------------------------ #
    # Main evaluation loop                                                 #
    # ------------------------------------------------------------------ #

    def test_dataset(self, dataset_paths, kernel_path):
        """Evaluate the model on every (dataset, kernel) combination."""
        kernels = load_levin_kernels(kernel_path)
        if self.args.kernel_indices is not None:
            kernels = [kernels[i] for i in self.args.kernel_indices]
            print(f"\nTesting with kernels {self.args.kernel_indices}")
        else:
            print(f"\nTesting with {len(kernels)} Levin kernels")

        loaded_num_channels = None
        all_dataset_aggs = {}

        for dataset_path in dataset_paths:
            image_files = self._list_images(dataset_path)
            if not image_files:
                print(f'No images found in {dataset_path}, skipping.')
                continue

            try:
                is_color = is_rgb(image_files[0])
            except Exception:
                raise RuntimeError(f'Could not open images in {dataset_path}')

            num_channels = 1 if self.args.gray else (3 if is_color else 1)
            color_mode = 'Grayscale' if num_channels == 1 else 'RGB'
            print(f'\n{color_mode} deblurring — {dataset_path}')

            if num_channels != loaded_num_channels:
                self._load_model(num_channels)
                loaded_num_channels = num_channels

            # Load all images once; kernels are the inner loop
            images = []
            for fp in image_files:
                img = self._load_image(fp)
                if num_channels == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
                images.append((os.path.splitext(os.path.basename(fp))[0], img))

            kernel_aggs = []
            for k_idx, kernel in enumerate(kernels):
                agg = dict(psnr=0., psnr_blurry=0., ssim=0., lpips=0., count=0)

                for img_stem, img in images:
                    deblurred, clean_t, blurry_t, runtime = self._run_model(img, kernel)
                    metrics = self._compute_metrics(deblurred, clean_t, blurry_t)

                    agg['psnr']        += metrics['psnr']
                    agg['psnr_blurry'] += metrics['psnr_blurry']
                    agg['ssim']        += metrics['ssim']
                    agg['lpips']       += metrics['lpips']
                    agg['count']       += 1

                    self._log_image(img_stem, k_idx, metrics, runtime)
                    if self.args.save or self.args.save_blurry:
                        save_dir = os.path.join('tests', self.args.save_dir)
                        if self.args.save:
                            out_name = f"{img_stem}_k{k_idx:02d}_deblurred.png"
                            cv2.imwrite(os.path.join(save_dir, out_name),
                                        metrics['deblurred_cv2'])
                        if self.args.save_blurry:
                            blurry_cv2 = variable_to_cv2_image(blurry_t)
                            out_name = f"{img_stem}_k{k_idx:02d}_blurry.png"
                            cv2.imwrite(os.path.join(save_dir, out_name), blurry_cv2)

                self._log_kernel_summary(dataset_path, k_idx, agg)
                kernel_aggs.append(agg)

            self._log_dataset_summary(dataset_path, kernel_aggs)
            all_dataset_aggs[dataset_path] = kernel_aggs

        if len(all_dataset_aggs) > 1:
            self._log_global_summary(all_dataset_aggs)


def test_deblur_dataset(args):
    """Entry point called by launcher_test.py when --task deblur is set."""
    dataset_paths = args.inputs if args.inputs is not None else [args.input]
    DatasetDeblurTester(args).test_dataset(dataset_paths, args.kernel_path)
