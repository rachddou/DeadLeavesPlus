"""
Test an inpainting model on multiple datasets, reporting scores for every
fixed p_keep value individually and as an aggregate.

The model expects C+1 input channels: cat([masked_image, mask_1c], dim=1).
Missing pixels are zero-filled; mask_1c is 1 where the pixel is observed.
"""
import os
from time import time
import numpy as np
import cv2
import torch
import lpips

from utils.models import UNetDeblur, UNetRes
from utils.utils import (init_logger_ipol, variable_to_cv2_image,
                         remove_dataparallel_wrapper, batch_ssim, is_rgb)
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Default p_keep values used when none are provided via args
DEFAULT_P_KEEP = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


class DatasetInpaintTester:

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
        net_in_ch = num_channels + 1   # image channels + 1-channel binary mask
        if args.model == 'UNetDeblur':
            network = UNetDeblur(in_nc=net_in_ch, out_nc=num_channels,
                                 nc=(64, 128, 256, 512))
        elif args.model == 'UNetRes':
            network = UNetRes(in_nc=net_in_ch, out_nc=num_channels,
                              nc=[64, 128, 256, 512], nb=4)
        else:
            raise ValueError(f"Unknown inpaint model: {args.model}. "
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

    def _make_mask(self, shape, p_keep, seed):
        """Generate a reproducible Bernoulli mask of shape (1, 1, H, W)."""
        rng = torch.Generator()
        rng.manual_seed(seed)
        mask = (torch.rand(shape, generator=rng) < p_keep).float()
        return mask

    def _run_model(self, clean_img, p_keep, img_seed):
        """Apply Bernoulli mask to clean_img, run model.

        Returns (inpainted_tensor, clean_tensor, masked_tensor, runtime).
        """
        clean_tensor, pad_h, pad_w = self._prepare_tensor(clean_img)

        _, C, H, W = clean_tensor.shape
        mask_1c = self._make_mask((1, 1, H, W), p_keep, seed=img_seed)

        with torch.no_grad():
            clean_t  = clean_tensor.type(self.dtype)
            mask_t   = mask_1c.type(self.dtype)
            masked_t = clean_t * mask_t               # zero-fill missing pixels
            inp      = torch.cat([masked_t, mask_t], dim=1)   # B × (C+1) × H × W

        start_time = time()
        with torch.no_grad():
            raw_out = self.model(inp)
        runtime = time() - start_time

        inpainted = torch.clamp(raw_out, 0., 1.)

        if pad_h:
            clean_t   = clean_t[:, :, :-1, :]
            inpainted = inpainted[:, :, :-1, :]
            masked_t  = masked_t[:, :, :-1, :]
        if pad_w:
            clean_t   = clean_t[:, :, :, :-1]
            inpainted = inpainted[:, :, :, :-1]
            masked_t  = masked_t[:, :, :, :-1]

        return inpainted, clean_t, masked_t, runtime

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #

    def _compute_metrics(self, inpainted_tensor, clean_tensor, masked_tensor):
        inpainted_cv2 = variable_to_cv2_image(inpainted_tensor)
        clean_cv2     = variable_to_cv2_image(clean_tensor)

        inpainted_f = np.asarray(inpainted_cv2, dtype=np.float32)
        clean_f     = np.asarray(clean_cv2, dtype=np.float32)

        mse  = np.mean((inpainted_f - clean_f) ** 2)
        psnr = 10 * np.log10((255 ** 2) / (mse + 1e-8))
        psnr_masked = compare_psnr(clean_cv2, variable_to_cv2_image(masked_tensor))
        ssim_val = batch_ssim(inpainted_tensor, clean_tensor)

        def _to_lpips(img_uint8):
            t = torch.from_numpy(img_uint8.transpose(2, 0, 1)).unsqueeze(0).float() / 127.5 - 1.0
            return t.cuda() if self.args.cuda else t

        lpips_val = self.lpips_fn(
            _to_lpips(inpainted_cv2), _to_lpips(clean_cv2)
        ).item()

        return dict(psnr=psnr, psnr_masked=psnr_masked, ssim=ssim_val,
                    lpips=lpips_val, inpainted_cv2=inpainted_cv2)

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #

    def _log_image(self, image_stem, p_keep, metrics, runtime):
        print(f"  [{image_stem}] p_keep={p_keep:.2f} | "
              f"PSNR={metrics['psnr']:.2f} dB | "
              f"LPIPS={metrics['lpips']:.4f} | t={runtime:.3f}s")
        self.logger.info(f"\t{image_stem} | p_keep={p_keep:.2f} | "
                         f"PSNR={metrics['psnr']:.4f} | "
                         f"PSNR_masked={metrics['psnr_masked']:.4f} | "
                         f"SSIM={metrics['ssim']:.4f} | "
                         f"LPIPS={metrics['lpips']:.4f} | "
                         f"t={runtime:.4f}s")

    def _log_pkeep_summary(self, dataset_path, p_keep, agg):
        n = agg['count']
        msg = (f"\n--- {dataset_path} | p_keep={p_keep:.2f} | {n} images ---\n"
               f"  PSNR        : {agg['psnr']/n:.4f} dB\n"
               f"  PSNR Masked : {agg['psnr_masked']/n:.4f} dB\n"
               f"  SSIM        : {agg['ssim']/n:.4f}\n"
               f"  LPIPS       : {agg['lpips']/n:.4f}")
        print(msg)
        self.logger.info(msg)

    def _log_dataset_summary(self, dataset_path, p_keep_values, all_pkeep_aggs):
        total = {k: 0.0 for k in ('psnr', 'psnr_masked', 'ssim', 'lpips', 'count')}
        for agg in all_pkeep_aggs:
            for k in total:
                total[k] += agg[k]
        n = total['count']
        msg = (f"\n=== {dataset_path} | ALL {len(p_keep_values)} p_keep values | {n} image-mask pairs ===\n"
               f"  PSNR        : {total['psnr']/n:.4f} dB\n"
               f"  PSNR Masked : {total['psnr_masked']/n:.4f} dB\n"
               f"  SSIM        : {total['ssim']/n:.4f}\n"
               f"  LPIPS       : {total['lpips']/n:.4f}")
        print(msg)
        self.logger.info(msg)

    def _log_global_summary(self, all_dataset_aggs):
        total = {k: 0.0 for k in ('psnr', 'psnr_masked', 'ssim', 'lpips', 'count')}
        for aggs in all_dataset_aggs.values():
            for agg in aggs:
                for k in total:
                    total[k] += agg[k]
        n = total['count']
        msg = (f"\n{'='*60}\n"
               f"GLOBAL SUMMARY — all datasets, all p_keep values — {n} pairs\n"
               f"  PSNR        : {total['psnr']/n:.4f} dB\n"
               f"  PSNR Masked : {total['psnr_masked']/n:.4f} dB\n"
               f"  SSIM        : {total['ssim']/n:.4f}\n"
               f"  LPIPS       : {total['lpips']/n:.4f}\n"
               f"{'='*60}")
        print(msg)
        self.logger.info(msg)

    # ------------------------------------------------------------------ #
    # Main evaluation loop                                                 #
    # ------------------------------------------------------------------ #

    def test_dataset(self, dataset_paths, p_keep_values):
        """Evaluate the model on every (dataset, p_keep) combination."""
        print(f"\nTesting with p_keep values: {p_keep_values}")

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
            print(f'\n{color_mode} inpainting — {dataset_path}')

            if num_channels != loaded_num_channels:
                self._load_model(num_channels)
                loaded_num_channels = num_channels

            images = []
            for fp in image_files:
                img = self._load_image(fp)
                if num_channels == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
                images.append((os.path.splitext(os.path.basename(fp))[0], img))

            pkeep_aggs = []
            for p_keep in p_keep_values:
                agg = dict(psnr=0., psnr_masked=0., ssim=0., lpips=0., count=0)

                for img_idx, (img_stem, img) in enumerate(images):
                    # Deterministic seed per (p_keep, image) pair
                    seed = int(p_keep * 1000) * 10000 + img_idx

                    inpainted, clean_t, masked_t, runtime = self._run_model(
                        img, p_keep, seed)
                    metrics = self._compute_metrics(inpainted, clean_t, masked_t)

                    agg['psnr']        += metrics['psnr']
                    agg['psnr_masked'] += metrics['psnr_masked']
                    agg['ssim']        += metrics['ssim']
                    agg['lpips']       += metrics['lpips']
                    agg['count']       += 1

                    self._log_image(img_stem, p_keep, metrics, runtime)
                    if self.args.save:
                        save_dir = os.path.join('tests', self.args.save_dir)
                        out_name = f"{img_stem}_p{int(p_keep*100):03d}_inpainted.png"
                        cv2.imwrite(os.path.join(save_dir, out_name),
                                    metrics['inpainted_cv2'])

                self._log_pkeep_summary(dataset_path, p_keep, agg)
                pkeep_aggs.append(agg)

            self._log_dataset_summary(dataset_path, p_keep_values, pkeep_aggs)
            all_dataset_aggs[dataset_path] = pkeep_aggs

        if len(all_dataset_aggs) > 1:
            self._log_global_summary(all_dataset_aggs)


def test_inpaint_dataset(args):
    """Entry point called by launcher_test.py when --task inpaint is set."""
    dataset_paths  = args.inputs if args.inputs is not None else [args.input]
    p_keep_values  = args.p_keep_values if args.p_keep_values else DEFAULT_P_KEEP
    DatasetInpaintTester(args).test_dataset(dataset_paths, p_keep_values)
