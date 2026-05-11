"""
Test a denoising model on multiple datasets and noise levels.
"""
import os
from time import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import lpips
from utils.models import FFDNet, UNetRes
from skimage.transform import pyramid_reduce, rotate
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.utils import (batch_ssim, normalize, init_logger_ipol,
                         variable_to_cv2_image, remove_dataparallel_wrapper,
                         is_rgb, compute_noise_map, rgb_2_grey_tensor)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class DatasetTester:

    def __init__(self, args):
        torch.manual_seed(0)
        self.args = args
        self.logger = init_logger_ipol(args.model_path.split('/')[-1].split('.')[0])
        self.dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
        self.lpips_fn = (lpips.LPIPS(net='alex').cuda()
                         if args.cuda else lpips.LPIPS(net='alex'))
        self.model = None
        self.noise_map_scale = None

    # ------------------------------------------------------------------ #
    # Model loading                                                        #
    # ------------------------------------------------------------------ #

    def _load_model(self, num_channels):
        args = self.args
        if args.model == 'FFDNet':
            network = FFDNet(num_input_channels=num_channels)
            self.noise_map_scale = 2
        elif args.model == 'DRUNet':
            self.noise_map_scale = 1
            if args.blind:
                network = UNetRes(in_nc=num_channels, out_nc=num_channels)
            else:
                network = UNetRes(in_nc=num_channels + 1, out_nc=num_channels)

        model_filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                      args.model_path)
        if args.cuda:
            state_dict = (torch.load(model_filepath)['state_dict']
                          if 'ckpt' in model_filepath else torch.load(model_filepath))
            self.model = (nn.DataParallel(network).cuda()
                          if args.model == 'FFDNet' else network.cuda())
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
        """Read one image, normalize to [0, 1] float32, crop to 1000×1000."""
        image = cv2.imread(filepath)
        if image is None:
            raise RuntimeError(f'Could not read {filepath}')
        image = normalize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if self.args.scale_invariance:
            image = pyramid_reduce(image, 4.5, channel_axis=2)
        return image[:800, :800, :]

    # ------------------------------------------------------------------ #
    # Distortions                                                          #
    # ------------------------------------------------------------------ #

    def _apply_distortions(self, image):
        """Apply geometric and photometric distortions.

        Returns (original_tensor, rotation_mask_or_None, pad_height, pad_width).
        """
        args = self.args
        rotation_mask = None

        if args.ud_invariance:
            image = np.flipud(image)
        if args.lr_invariance:
            image = np.fliplr(image)
        if args.rot_invariance:
            rotation_angle = args.angle
            image = rotate(image, rotation_angle, resize=True,
                           preserve_range=True, mode='constant')
            rotation_mask = np.bool_(rotate(np.ones(image.shape), rotation_angle,
                                            resize=True, preserve_range=True,
                                            mode='constant'))

        crop_height = 8 * (image.shape[0] // 8)
        crop_width = 8 * (image.shape[1] // 8)
        image = image.transpose(2, 0, 1)[:, :crop_height, :crop_width]
        if rotation_mask is not None:
            rotation_mask = rotation_mask[:crop_height, :crop_width, :]

        image = np.expand_dims(image, 0)

        # Pad to even spatial dimensions (required by some architectures)
        pad_height = image.shape[2] % 2 == 1
        pad_width = image.shape[3] % 2 == 1
        if pad_height:
            image = np.concatenate((image, image[:, :, -1:, :]), axis=2)
        if pad_width:
            image = np.concatenate((image, image[:, :, :, -1:]), axis=3)

        if args.contrast_reduction:
            contrast_scale = 0.1
            for i in range(image.shape[1]):
                channel_min, channel_max = image[:, i].min(), image[:, i].max()
                image[:, i] = contrast_scale * (image[:, i] - channel_min) / (1e-8 + channel_max - channel_min)
                image[:, i] += 0.5 - image[:, i].mean()
            image = np.clip(image, 0, 1)

        return torch.Tensor(image.copy()), rotation_mask, pad_height, pad_width

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def _run_model(self, original_tensor, noise_sigma, pad_height, pad_width):
        """Add noise, run model, trim padding.

        Returns (denoised_tensor, clean_tensor, noisy_tensor, runtime).
        """
        args = self.args

        if args.add_noise:
            noise = torch.FloatTensor(original_tensor.size()).normal_(mean=0, std=noise_sigma)
            noisy = original_tensor + noise
        else:
            noisy = original_tensor.clone()

        with torch.no_grad():
            clean_tensor = Variable(original_tensor.type(self.dtype))
            noisy_tensor = Variable(noisy.type(self.dtype))
            sigma_tensor = Variable(torch.FloatTensor([noise_sigma]).type(self.dtype))

        start_time = time()
        if not args.blind:
            noise_map = compute_noise_map(noisy_tensor, sigma_tensor,
                                          self.noise_map_scale, mode=args.noise_map)
            if args.model == 'FFDNet':
                raw_output = self.model(noisy_tensor, noise_map)
            else:
                raw_output = self.model(torch.cat((noisy_tensor, noise_map), dim=1))
        else:
            raw_output = self.model(noisy_tensor)
        runtime = time() - start_time

        denoised_tensor = torch.clamp(
            noisy_tensor - raw_output if args.residual else raw_output, 0., 1.)

        if pad_height:
            clean_tensor = clean_tensor[:, :, :-1, :]
            denoised_tensor = denoised_tensor[:, :, :-1, :]
        if pad_width:
            clean_tensor = clean_tensor[:, :, :, :-1]
            denoised_tensor = denoised_tensor[:, :, :, :-1]

        return denoised_tensor, clean_tensor, noisy_tensor, runtime

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #

    def _compute_metrics(self, denoised_tensor, clean_tensor, noisy_tensor, rotation_mask):
        """Compute PSNR, PSNR-noisy, SSIM and LPIPS. Returns a metrics dict."""
        args = self.args
        denoised_image = np.float64(variable_to_cv2_image(denoised_tensor))

        if args.contrast_reduction:
            contrast_scale = 0.1
            for i in range(denoised_image.shape[2]):
                denoised_image[..., i] = (
                    (1. / contrast_scale) * (denoised_image[..., i] - denoised_image[..., i].min())
                )

        original_image = variable_to_cv2_image(clean_tensor)
        residual = original_image - denoised_image
        residual = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8) * 255

        denoised_image = np.uint8(np.clip(denoised_image, 0, 255))
        denoised_float = np.asarray(denoised_image, dtype=np.float32)
        original_float = np.asarray(original_image, dtype=np.float32)

        if args.rot_invariance and rotation_mask is not None:
            mse = np.sum(rotation_mask * (denoised_float - original_float) ** 2) / (np.sum(rotation_mask) + 1e-8)
        else:
            mse = np.mean((denoised_float - original_float) ** 2)
        psnr = 10 * np.log10((255 ** 2) / (mse + 1e-8))
        psnr_noisy = compare_psnr(variable_to_cv2_image(noisy_tensor), original_image)
        ssim_val = batch_ssim(denoised_tensor, clean_tensor) if args.add_noise else 0.0

        def _to_lpips_tensor(img_uint8):
            tensor = torch.from_numpy(img_uint8.transpose(2, 0, 1)).unsqueeze(0).float() / 127.5 - 1.0
            return tensor.cuda() if args.cuda else tensor

        lpips_val = self.lpips_fn(
            _to_lpips_tensor(denoised_image), _to_lpips_tensor(original_image)
        ).item()

        return dict(psnr=psnr, psnr_noisy=psnr_noisy, ssim=ssim_val,
                    lpips=lpips_val, denoised_image=denoised_image, residual=residual)

    # ------------------------------------------------------------------ #
    # Result logging                                                       #
    # ------------------------------------------------------------------ #

    def _log_image(self, image_stem, metrics, runtime):
        print(f"  {image_stem} | PSNR={metrics['psnr']:.2f} dB | "
              f"LPIPS={metrics['lpips']:.4f} | t={runtime:.3f}s")
        self.logger.info(f"\t{image_stem} | Runtime {runtime:.4f}s")

    def _log_dataset_summary(self, dataset_path, noise_sigma, aggregated_metrics):
        num_images = aggregated_metrics['count']
        summary = (
            f"\n=== {dataset_path} | sigma={noise_sigma:.3f} | {num_images} images ===\n"
            f"  PSNR        : {aggregated_metrics['psnr']/num_images:.4f} dB\n"
            f"  PSNR Noisy  : {aggregated_metrics['psnr_noisy']/num_images:.4f} dB\n"
            f"  SSIM        : {aggregated_metrics['ssim']/num_images:.4f}\n"
            f"  LPIPS       : {aggregated_metrics['lpips']/num_images:.4f}"
        )
        print(summary)
        self.logger.info(summary)

    def _save_outputs(self, image_stem, metrics):
        args = self.args
        save_dir = os.path.join('tests', args.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, image_stem + '_denoised.png'), metrics['denoised_image'])
        if args.save_res:
            cv2.imwrite(os.path.join(save_dir, image_stem + '_residual.png'), metrics['residual'])

    # ------------------------------------------------------------------ #
    # Main evaluation loop                                                 #
    # ------------------------------------------------------------------ #

    def test_dataset(self, dataset_paths, noise_sigmas):
        """Evaluate the model on every (dataset, noise_sigma) combination."""
        args = self.args
        loaded_num_channels = None

        for dataset_path in dataset_paths:
            image_files = self._list_images(dataset_path)
            if not image_files:
                print(f'No images found in {dataset_path}, skipping.')
                continue

            try:
                is_color = is_rgb(image_files[0])
            except Exception:
                raise RuntimeError(f'Could not open images in {dataset_path}')

            num_channels = 3 if is_color else 1
            color_mode = 'RGB' if is_color else 'Grayscale'
            print(f'\n{color_mode} denoising — {dataset_path}')

            if num_channels != loaded_num_channels:
                self._load_model(num_channels)
                loaded_num_channels = num_channels

            for noise_sigma in noise_sigmas:
                aggregated_metrics = dict(psnr=0., psnr_noisy=0., ssim=0., lpips=0., count=0)

                for filepath in image_files:
                    image_stem = os.path.splitext(os.path.basename(filepath))[0]
                    image = self._load_image(filepath)
                    original_tensor, rotation_mask, pad_height, pad_width = self._apply_distortions(image)
                    denoised_tensor, clean_tensor, noisy_tensor, runtime = self._run_model(
                        original_tensor, noise_sigma, pad_height, pad_width)
                    metrics = self._compute_metrics(
                        denoised_tensor, clean_tensor, noisy_tensor, rotation_mask)

                    aggregated_metrics['psnr'] += metrics['psnr']
                    aggregated_metrics['psnr_noisy'] += metrics['psnr_noisy']
                    aggregated_metrics['ssim'] += metrics['ssim']
                    aggregated_metrics['lpips'] += metrics['lpips']
                    aggregated_metrics['count'] += 1

                    self._log_image(image_stem, metrics, runtime)
                    if args.save:
                        self._save_outputs(image_stem, metrics)

                self._log_dataset_summary(dataset_path, noise_sigma, aggregated_metrics)


def test_dataset(args):
    """Entry point. Accepts 'inputs' (list of paths) and 'noise_sigmas' (list of floats),
    falling back to the single 'input' / 'noise_sigma' args for backward compatibility."""
    dataset_paths = args.inputs if args.inputs is not None else [args.input]
    noise_sigmas = args.noise_sigmas if args.noise_sigmas is not None else [args.noise_sigma]
    DatasetTester(args).test_dataset(dataset_paths, noise_sigmas)
