"""GPU-accelerated dead leaves image generator.

Parallelism strategy
--------------------
* **Across images** – a ``batch_size > 1`` generates multiple independent images
  simultaneously; the stacking loop places one shape at a time but the occlusion
  update and compositing are fully vectorised across the whole batch on GPU.
* **Texture generation** – ``generate_textures_dictionary`` pre-computes all N
  textures in one GPU batch (using batched FFT in ``colored_noise``).
* **DoF compositing** – blurring and alpha compositing are executed on GPU tensors.

Shape generation (polygon, disk, rectangle) remains on CPU because it relies on
``shapely`` / ``rasterio`` which have no GPU equivalent.
"""
import logging
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
import skimage.io as skio
import torch
import torch.nn.functional as F

from dead_leaves_generation_pytorch.polygons_maker_bis import (
    binary_polygon_generator,
    make_rectangle_mask,
)
from dead_leaves_generation_pytorch.utils.colored_noise import sample_color_noise_batch
from dead_leaves_generation_pytorch.utils.interpolation_maps import sample_interpolation_map
from dead_leaves_generation_pytorch.utils.perspective import perspective_shift
from dead_leaves_generation_pytorch.utils.texture_generation import (
    bilevelTextureMixer_batch,
    _sample_slope_from_ranges,
)
from dead_leaves_generation_pytorch.utils.utils import (
    linear_color_gradient_torch,
    normalize_torch,
    rgb_to_lab_torch,
    lab_to_rgb_torch,
    rotate_torch,
)

logger = logging.getLogger(__name__)

_NPY_PATH = Path(__file__).parent.parent / "npy" / "dict.npy"
dict_instance = np.load(_NPY_PATH, allow_pickle=True)


# ---------------------------------------------------------------------------
# Gaussian blur helper (no extra dependency needed)
# ---------------------------------------------------------------------------

def _gaussian_blur_batch(images: torch.Tensor, sigma: float, kernel_size: int = 11) -> torch.Tensor:
    """Apply Gaussian blur to a (B, C, H, W) float tensor."""
    device = images.device
    ks = kernel_size
    x = torch.arange(ks, dtype=torch.float32, device=device) - ks // 2
    k1d = torch.exp(-0.5 * (x / sigma) ** 2)
    k1d = k1d / k1d.sum()
    k2d = k1d.outer(k1d).unsqueeze(0).unsqueeze(0)  # (1,1,ks,ks)
    C = images.shape[1]
    k2d = k2d.expand(C, 1, -1, -1)                  # (C,1,ks,ks)
    pad = ks // 2
    padded = F.pad(images, [pad, pad, pad, pad], mode='reflect')
    return F.conv2d(padded, k2d, groups=C)


# ---------------------------------------------------------------------------
# Textures – GPU version of the texture-only generator
# ---------------------------------------------------------------------------

class Textures:
    """GPU texture generator (direct port of the CPU Textures class).

    Parameters
    ----------
    width : int
        Output image size.
    use_natural_images : bool
        Sample colours from real images.
    path : str
        Directory containing source images (required when use_natural_images=True).
    types : list[str]
        Texture types to sample from; subset of
        ``["sin", "freq_noise", "texture_mixes", "gradient"]``.
    type_weights : list[float]
        Sampling probabilities for each type (must sum to 1).
    slope_range : list
        1/f frequency slope range; ``[a, b]`` or ``[[a1,b1], ...]``.
    apply_warp : bool
        Apply geometric warp to interpolation maps.
    random_phase : bool
        Randomise FFT phase for gradient textures.
    device : torch.device or None
        Computation device; defaults to CUDA if available else CPU.
    """

    def __init__(
        self,
        width: int = 512,
        use_natural_images: bool = True,
        path: str = "",
        types: list = None,
        type_weights: list = None,
        slope_range=None,
        apply_warp: bool = True,
        random_phase: bool = False,
        device: torch.device = None,
    ):
        if types is None:
            types = ["sin"]
        if type_weights is None:
            type_weights = [1.0]
        if slope_range is None:
            slope_range = [0.5, 2.5]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.width = width
        self.use_natural_images = use_natural_images
        self.path = path
        self.apply_warp = apply_warp
        self.random_phase = random_phase
        self.online_generation = True

        self.files: list = []
        self.img_source = np.zeros((100, 100, 3), dtype=np.uint8)
        self.src_h, self.src_w = 100, 100
        # GPU copy of the source image – updated every time source_image_sampling runs
        self.img_source_t = torch.zeros(100, 100, 3, dtype=torch.uint8, device=self.device)

        self.resulting_image = torch.ones(width, width, 3, dtype=torch.uint8, device=self.device)
        self.perspective_shift_flag = False
        self.perspective_var = True

        if self.use_natural_images:
            self.source_image_sampling()

        self.texture_type_lists = types
        self.texture_type_frequency = type_weights
        self.slope_range = slope_range
        self.texture_type = types[0]
        self.sample_texture_type()

    # ------------------------------------------------------------------
    def _sample_slope(self) -> float:
        return _sample_slope_from_ranges(self.slope_range)

    def random_patch_selection(self) -> torch.Tensor:
        """Return a random 100×100 patch from the source image as a GPU tensor."""
        x = torch.randint(0, max(1, self.src_h - 101), ()).item()
        y = torch.randint(0, max(1, self.src_w - 101), ()).item()
        return self.img_source_t[x:x + 100, y:y + 100]   # (100, 100, 3) uint8 on device

    def source_image_sampling(self):
        """Load a random 3-channel source image from self.path."""
        if not self.files:
            self.files = [
                os.path.join(self.path, f)
                for f in os.listdir(self.path)
                if os.path.isfile(os.path.join(self.path, f))
            ]
        ind = random.randint(0, len(self.files) - 1)
        self.img_source = skio.imread(self.files[ind])
        self.src_h, self.src_w = self.img_source.shape[:2]
        self.img_source_t = torch.from_numpy(
            np.ascontiguousarray(self.img_source)).to(self.device)
        logger.debug("Loaded source image: %s", self.files[ind])

    def sample_texture_type(self):
        """Sample a texture type according to type_weights."""
        if len(self.texture_type_lists) == 1:
            self.texture_type = self.texture_type_lists[0]
        else:
            weights = torch.tensor(self.texture_type_frequency, dtype=torch.float32)
            idx = torch.multinomial(weights, 1).item()
            self.texture_type = self.texture_type_lists[idx]

    def _generate_source_texture(self, width: int) -> torch.Tensor:
        """Generate one texture map of size (3, width, width) uint8 on device."""
        if self.texture_type == "freq_noise":
            slope = self._sample_slope()
            batch = sample_color_noise_batch(
                [self.random_patch_selection()], width, [slope], self.device)
            return batch[0]  # (3, W, W) uint8

        if self.texture_type == "texture_mixes":
            singleColor1 = torch.rand(()).item() < 0.1
            singleColor2 = torch.rand(()).item() < 0.1
            mix_modes = [["sin"], ["grid"]]
            mode = mix_modes[0] if torch.rand(()).item() < 0.9 else mix_modes[1]
        else:
            singleColor1 = True
            singleColor2 = True
            mode = [self.texture_type]

        warp = torch.rand(()).item() < 0.5 if self.apply_warp else False
        thresh = torch.randint(5, 50, ()).item()
        batch = bilevelTextureMixer_batch(
            [self.random_patch_selection()],
            [self.random_patch_selection()],
            single_color1=singleColor1, single_color2=singleColor2,
            mixing_types=mode, width=width,
            thresh_val=thresh, warp=warp,
            slope_range=self.slope_range, device=self.device)
        return batch[0]  # (3, W, W) uint8

    def generate_texture(self, width: int = 100) -> torch.Tensor:
        """Generate one texture map.

        Returns:
            (3, width, width) uint8 tensor on self.device.
        """
        self.sample_texture_type()
        tmp_random_phase = self.random_phase
        logger.debug("Texture type: %s", self.texture_type)

        if self.texture_type == "gradient":
            px = torch.randint(0, self.src_h, ()).item()
            py = torch.randint(0, self.src_w, ()).item()
            c1 = self.img_source_t[px, py].to(self.device)
            px2 = torch.randint(0, self.src_h, ()).item()
            py2 = torch.randint(0, self.src_w, ()).item()
            c2 = self.img_source_t[px2, py2].to(self.device)
            k = float(torch.empty(()).uniform_(0.1, 0.5).item())
            out = linear_color_gradient_torch(c1, c2, width, 45, k, self.device)
            return out.permute(2, 0, 1)  # (3, W, W)

        if self.texture_type in ["grid", "texture_mixes"] and self.random_phase:
            self.random_phase = False
        self.perspective_var = torch.rand(()).item() > 0.5
        if self.perspective_var and self.perspective_shift_flag:
            logger.debug("Applying perspective shift")
            tex = self._generate_source_texture(2 * width)           # (3,2W,2W)
            tex_hwc = tex.permute(1, 2, 0)                           # (2W,2W,3)
            tex_hwc = perspective_shift(tex_hwc)                     # cropped
            tex = tex_hwc.permute(2, 0, 1)                           # (3,H,W)
        else:
            tex = self._generate_source_texture(width)
        self.random_phase = tmp_random_phase
        return tex  # (3, W, W) uint8

    def clear(self):
        self.resulting_image = torch.ones(
            self.width, self.width, 3, dtype=torch.uint8, device=self.device)


# ---------------------------------------------------------------------------
# Deadleaves – batch GPU generator
# ---------------------------------------------------------------------------

class Deadleaves(Textures):
    """Batch GPU dead-leaves image generator.

    Inherits texture generation from :class:`Textures` and adds multi-layer
    depth-of-field compositing and batch processing across ``batch_size`` images.

    Parameters
    ----------
    rmin, rmax : int
        Minimum / maximum shape radius in pixels.
    power_law_exponent : float
        Exponent of the shape-size power law (higher → more small shapes).
    width : int
        Output image size in pixels (square).
    batch_size : int
        Number of images to generate in parallel.
    device : torch.device or None
        Computation device.
    (other parameters documented in :class:`Textures`)
    """

    def __init__(
        self,
        rmin: int = 1,
        rmax: int = 1000,
        power_law_exponent: float = 3.0,
        width: int = 512,
        use_natural_images: bool = True,
        path: str = "",
        texture_path: str = "",
        shape_type: str = "poly",
        types: list = None,
        type_weights: list = None,
        slope_range=None,
        use_texture: bool = True,
        online_generation: bool = False,
        apply_warp: bool = True,
        random_phase: bool = False,
        perspective: bool = True,
        device: torch.device = None,
        batch_size: int = 1,
    ):
        if types is None:
            types = ["sin"]
        if type_weights is None:
            type_weights = [1.0]
        if slope_range is None:
            slope_range = [0.5, 2.5]

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # Texture parent attributes (manual init – no auto source_image_sampling)
        self.width = width
        self.use_natural_images = use_natural_images
        self.path = path
        self.apply_warp = apply_warp
        self.random_phase = random_phase
        self.online_generation = online_generation
        self.texture_path = texture_path
        self.shape_type = shape_type
        self.use_texture = use_texture
        self.texture_type_lists = types
        self.texture_type_frequency = type_weights
        self.slope_range = slope_range
        self.texture_type = types[0]
        self.sample_texture_type()
        self.perspective_shift_flag = perspective
        self.perspective_var = True

        self.files: list = []
        self.img_source = np.zeros((100, 100, 3), dtype=np.uint8)
        self.src_h, self.src_w = 100, 100
        self.img_source_t = torch.zeros(100, 100, 3, dtype=torch.uint8, device=self.device)

        self.n_textures = 10
        self.textures: list = []        # list of dicts {"1": tensor, "2": tensor, ...}

        # Shape distribution parameters
        self.rmin = rmin
        self.rmax = rmax
        self.power_law_exponent = power_law_exponent
        self.vamin = 1.0 / (rmax ** (power_law_exponent - 1))
        self.vamax = 1.0 / (rmin ** (power_law_exponent - 1))

        def _theoretical_n(delta, r_min, r_max, w, alpha):
            d = 0.0
            e = alpha - 1
            for k in range(r_min, r_max):
                d += (1.0 / ((1.0 / r_min ** e) - (1.0 / r_max ** e))
                      * (1.0 - (k / (k + 1)) ** 2))
            d *= math.pi / w ** 2
            if d >= 1.0:
                return 20.0
            return math.log(delta) / math.log(1.0 - d)

        n_shapes = int(_theoretical_n(0.002, rmin, rmax, width, power_law_exponent))
        interval = max(1, n_shapes // 10 if n_shapes > 10 else n_shapes)
        self.interval = interval
        logger.debug("Expected shapes per layer: %d", interval)

        # Batch GPU state
        self._init_batch_state()

    # ------------------------------------------------------------------
    def _init_batch_state(self):
        B, W = self.batch_size, self.width
        # Occlusion mask on GPU: True = pixel still uncovered
        self.binary_images = torch.ones(B, W, W, dtype=torch.bool, device=self.device)
        # Compositing canvas on GPU
        self.resulting_images = torch.ones(
            B, 3, W, W, dtype=torch.float32, device=self.device)

    # expose a single resulting_image for API compatibility
    @property
    def resulting_image(self) -> torch.Tensor:
        return self.resulting_images[0].permute(1, 2, 0)  # (H, W, 3)

    @resulting_image.setter
    def resulting_image(self, v):
        pass  # set via _init_batch_state

    # ------------------------------------------------------------------
    def update_leaves_size_parameters(self, rmin: int, rmax: int):
        self.rmin = rmin
        self.rmax = rmax
        self.vamin = 1.0 / (rmax ** (self.power_law_exponent - 1))
        self.vamax = 1.0 / (rmin ** (self.power_law_exponent - 1))

    # ------------------------------------------------------------------
    def source_image_sampling(self):
        if not self.files:
            self.files = [
                os.path.join(self.path, f)
                for f in os.listdir(self.path)
                if os.path.isfile(os.path.join(self.path, f))
            ]
        ind = random.randint(0, len(self.files) - 1)
        self.img_source = skio.imread(self.files[ind])
        self.src_h, self.src_w = self.img_source.shape[:2]
        self.img_source_t = torch.from_numpy(
            np.ascontiguousarray(self.img_source)).to(self.device)
        logger.debug("Loaded source image: %s", self.files[ind])

    # ------------------------------------------------------------------
    def generate_textures_dictionary(self):
        """Pre-compute n_textures textures at 5 downscale levels on GPU."""
        tex_size = 2 * self.rmax + 100
        self.textures = []

        for _ in range(self.n_textures):
            self.sample_texture_type()
            tt = self.texture_type
            if tt == "texture_mixes":
                sc1 = torch.rand(()).item() < 0.1
                sc2 = torch.rand(()).item() < 0.1
                modes = [["sin"], ["grid"]]
                mode = modes[0] if torch.rand(()).item() < 0.9 else modes[1]
            else:
                sc1, sc2 = True, True
                mode = [tt] if tt != "freq_noise" else ["sin"]
            warp = torch.rand(()).item() < 0.5 if self.apply_warp else False
            thresh = torch.randint(5, 50, ()).item()

            t = bilevelTextureMixer_batch(
                [self.random_patch_selection()], [self.random_patch_selection()],
                single_color1=sc1, single_color2=sc2,
                mixing_types=mode, width=tex_size,
                thresh_val=thresh, warp=warp,
                slope_range=self.slope_range, device=self.device,
            )[0]  # (3, H, W) uint8

            scale_dict = {"1": t}
            h, w = t.shape[1], t.shape[2]
            tf = t.float().unsqueeze(0)  # (1, 3, H, W)
            for s in range(2, 6):
                scaled = F.interpolate(tf, size=(h // s, w // s), mode='area')
                scale_dict[str(s)] = scaled[0].to(torch.uint8)
            self.textures.append(scale_dict)

    # ------------------------------------------------------------------
    def fetch_textures(self):
        """Load textures from texture_path and upload to device."""
        files = [os.path.join(self.texture_path, f)
                 for f in os.listdir(self.texture_path)]
        ids = torch.randint(0, len(files), (self.n_textures,)).tolist()
        self.textures = []
        for idx in ids:
            img = skio.imread(files[idx])
            t = torch.from_numpy(img).to(self.device).permute(2, 0, 1)  # (3,H,W)
            scale_dict = {"1": t}
            h, w = t.shape[1], t.shape[2]
            tf = t.float().unsqueeze(0)
            for s in range(2, 6):
                scale_dict[str(s)] = F.interpolate(tf, size=(h // s, w // s),
                                                   mode='area')[0].to(torch.uint8)
            self.textures.append(scale_dict)

    # ------------------------------------------------------------------
    def _pick_texture(self, size: int) -> torch.Tensor:
        """Select a texture from the precomputed dictionary.

        Returns:
            (3, H, W) uint8 tensor on self.device.
        """
        tex_dict = self.textures[torch.randint(0, self.n_textures, ()).item()]
        h = tex_dict["1"].shape[1]
        max_scale = min(5, h / max(1, size))
        scale = int(math.floor(1 + max(0, (max_scale - 1) * float(
            torch.empty(()).uniform_(0, 1).pow(2 / 3).item()))))
        tex = tex_dict[str(scale)]
        transform = torch.randint(0, 4, ()).item()
        if transform == 0:
            tex = tex.flip(1)
        elif transform == 1:
            tex = tex.flip(2)
        elif transform == 2:
            tex = tex.rot90(1, [1, 2])
        elif transform == 3:
            tex = tex.flip(1).rot90(1, [1, 2])
        return tex

    # ------------------------------------------------------------------
    def generate_single_shape_mask(self):
        """Sample a shape from the power-law distribution (CPU).

        Returns:
            (shape_mask np.ndarray bool, radius int)
        """
        v = self.vamin + (self.vamax - self.vamin) * torch.rand(()).item()
        radius = int(1.0 / (v ** (1.0 / (self.power_law_exponent - 1))))
        shape = self.shape_type
        if shape == "mix":
            shape = random.choice(["disk", "poly", "rectangle"])

        if shape == "poly":
            shape_1d = binary_polygon_generator(
                2 * (3 * radius // 2) + 1,
                n=torch.randint(50, max(100, int(0.9 * radius)), ()).item(),
                allow_holes=bool(random.getrandbits(1)),
                smoothing=bool(random.getrandbits(1)))
            if max(shape_1d.shape) >= 2 * self.rmax + 100:
                sc = (2 * self.rmax + 100.0) / max(shape_1d.shape)
                ns = (int(2 * ((shape_1d.shape[0] * sc) // 2) - 1),
                      int(2 * ((shape_1d.shape[1] * sc) // 2) - 1))
                shape_1d = np.bool_(cv2.resize(np.uint8(shape_1d), ns,
                                               interpolation=cv2.INTER_AREA))
        elif shape == "disk":
            shape_1d = dict_instance[()][str(radius)]
        else:  # rectangle
            shape_1d = make_rectangle_mask(radius)
            if max(shape_1d.shape) >= 2 * self.rmax + 100:
                sc = (2 * self.rmax + 100.0) / max(shape_1d.shape)
                ns = (int(2 * ((shape_1d.shape[0] * sc) // 2) - 1),
                      int(2 * ((shape_1d.shape[1] * sc) // 2) - 1))
                shape_1d = np.bool_(cv2.resize(np.uint8(shape_1d), ns,
                                               interpolation=cv2.INTER_AREA))
        return shape_1d, radius

    # ------------------------------------------------------------------
    def _render_texture_gpu(self, mask_crop: torch.Tensor, radius: int) -> torch.Tensor:
        """Render a texture fill for a shape region entirely on GPU.

        Args:
            mask_crop: (h, w) float32 GPU tensor – the effective shape mask.
            radius: Shape radius (controls texture vs flat colour).

        Returns:
            (3, h, w) float32 GPU tensor.
        """
        ws, ls = mask_crop.shape
        shape_mask3 = mask_crop.unsqueeze(0).expand(3, -1, -1)  # (3, h, w)

        # Sample base colour from source image on GPU
        px = torch.randint(0, self.src_h, (), device=self.device)
        py = torch.randint(0, self.src_w, (), device=self.device)
        color = self.img_source_t[px, py].float()   # (3,)

        if radius < 20:
            return shape_mask3 * color.view(3, 1, 1)

        px2 = torch.randint(0, self.src_h, (), device=self.device)
        py2 = torch.randint(0, self.src_w, (), device=self.device)
        color2 = self.img_source_t[px2, py2].float()   # (3,)

        gvt = torch.rand((), device=self.device).item() if self.use_texture else 1.0

        if gvt > 0.95:
            k = float(torch.empty(()).uniform_(0.1, 0.5).item())
            angle = torch.randint(0, 360, ()).item()
            tex_hwc = linear_color_gradient_torch(
                color.to(torch.uint8), color2.to(torch.uint8),
                2 * max(ws, ls) + 1, angle, k, self.device)   # (W,W,3) uint8
            tex = tex_hwc.permute(2, 0, 1).float()             # (3,W,W)
        else:
            if self.online_generation:
                tex_chw = self.generate_texture(width=60 + max(ws, ls))  # (3,W,W)
                tex = tex_chw.float()
            else:
                tex = self._pick_texture(max(ws, ls)).float()

        th, tw = tex.shape[1], tex.shape[2]
        ox = torch.randint(0, max(1, th - ws), ()).item()
        oy = torch.randint(0, max(1, tw - ls), ()).item()
        tex_crop = tex[:, ox:ox + ws, oy:oy + ls]
        return shape_mask3 * tex_crop

    # ------------------------------------------------------------------
    def generate_stack_batch(self, disk_count: int):
        """Place disk_count shapes onto all batch images simultaneously.

        For each shape step:
        1. One shape geometry is generated on CPU and uploaded to GPU once.
        2. B random positions are sampled; the shape is placed at each position
           in a (B, W, W) boolean canvas on GPU.
        3. Occlusion update is vectorised across all B items in a single GPU op.
        4. Texture rendering and compositing loop over B items but touch only
           GPU tensors – no CPU↔GPU transfers inside the loop.

        Returns:
            tuple: (resulting_images (B,3,H,W) uint8, filled_mask (B,3,H,W) uint8)
        """
        B, W = self.batch_size, self.width

        for _ in range(disk_count):
            # --- CPU: generate one shape (shared geometry across all B images) ---
            shape_1d, radius = self.generate_single_shape_mask()
            ws, ls = shape_1d.shape

            # Upload shape mask to GPU once
            shape_bool = torch.from_numpy(
                np.ascontiguousarray(shape_1d)).to(dtype=torch.bool, device=self.device)

            # --- GPU: place shape at B random positions ---
            shape_canvas = torch.zeros(B, W, W, dtype=torch.bool, device=self.device)
            crop_bounds = []   # (xm, xM, ym, yM) per item

            px_all = torch.randint(0, W, (B,))
            py_all = torch.randint(0, W, (B,))

            for b in range(B):
                px, py = px_all[b].item(), py_all[b].item()
                xm = max(0, px - ws // 2)
                xM = min(W, 1 + px + ws // 2)
                ym = max(0, py - ls // 2)
                yM = min(W, 1 + py + ls // 2)
                # Corresponding crop inside the shape array
                sx = max(0, ws // 2 - px)
                ex = min(ws, W + ws // 2 - px)
                sy = max(0, ls // 2 - py)
                ey = min(ls, W + ls // 2 - py)
                if (xM > xm) and (yM > ym) and (ex > sx) and (ey > sy):
                    shape_canvas[b, xm:xM, ym:yM] = shape_bool[sx:ex, sy:ey]
                crop_bounds.append((xm, xM, ym, yM))

            # --- GPU: vectorised occlusion update for ALL B items at once ---
            effective = self.binary_images & shape_canvas   # (B, W, W) bool
            self.binary_images &= ~effective                # mark as covered

            # --- GPU: per-item texture render + composite (no CPU round-trips) ---
            for b in range(B):
                xm, xM, ym, yM = crop_bounds[b]
                eff_crop = effective[b, xm:xM, ym:yM].float()   # (h, w) on GPU
                if eff_crop.sum() == 0:
                    continue
                shape_render = self._render_texture_gpu(eff_crop, radius)  # (3, h, w)
                shape_mask3 = eff_crop.unsqueeze(0).expand(3, -1, -1)
                canvas = self.resulting_images[b, :, xm:xM, ym:yM]
                self.resulting_images[b, :, xm:xM, ym:yM] = (
                    canvas * (1.0 - shape_mask3) + shape_render
                ).clamp(0, 255)

        logger.debug("Stack created (%d shapes × %d images)", disk_count, B)
        result_u8 = self.resulting_images.to(torch.uint8)
        # filled_mask: 1 where pixels have been painted, 0 where background remains
        filled = (~self.binary_images).unsqueeze(1).expand(-1, 3, -1, -1).to(torch.uint8)
        return result_u8, filled

    # ------------------------------------------------------------------
    def clear(self):
        self._init_batch_state()

    # ------------------------------------------------------------------
    def compose_dead_leaves_depth_of_field_batch(
        self,
        blur_type: str,
        blur_val: float,
        fetch: bool = False,
    ):
        """Three-layer depth-of-field compositing for all batch items on GPU.

        Layers: background (blurred) → plainground (in focus) → foreground (blurred).
        """
        if self.use_natural_images:
            for _ in range(10):
                self.source_image_sampling()
                if self.img_source.ndim == 3 and self.img_source.shape[2] == 3:
                    break
            else:
                raise RuntimeError(
                    "Could not load a 3-channel source image after 10 attempts.")

        if self.use_texture and not self.online_generation:
            if fetch:
                self.fetch_textures()
            else:
                self.generate_textures_dictionary()

        # Background layer
        background, _ = self.generate_stack_batch(10 * self.interval)
        self.clear()
        # Plainground layer
        plainground, plain_mask = self.generate_stack_batch(self.interval)
        self.clear()
        # Foreground layer
        foreground, fore_mask = self.generate_stack_batch(max(1, int(0.5 * self.interval)))

        bg = background.float()           # (B,3,H,W)
        pg = plainground.float()
        fg = foreground.float()
        pm = plain_mask.float()           # (B,3,H,W)
        fm = fore_mask.float()

        if blur_type == "gaussian" and blur_val > 0:
            bg = _gaussian_blur_batch(bg, blur_val)

        pg_comp = (1.0 - pm) * bg + pm * pg

        fg_full = fg + (1.0 - fm) * pg_comp
        if blur_type == "gaussian" and blur_val > 0:
            fg_full = _gaussian_blur_batch(fg_full, blur_val)
            fm = _gaussian_blur_batch(fm, blur_val)

        result = pg_comp * (1.0 - fm) + fg_full * fm
        self.resulting_images = result.clamp(0, 255)

    def postprocess_batch(self, blur: bool = True, downscale: bool = True):
        """Post-process all batch images on GPU."""
        imgs = self.resulting_images  # (B,3,H,W) float
        if blur:
            sigma = float(torch.empty(()).uniform_(1, 3).item())
            imgs = _gaussian_blur_batch(imgs, sigma)
        if downscale:
            imgs = F.interpolate(imgs, scale_factor=0.5, mode='area',
                                 recompute_scale_factor=False)
        self.resulting_images = imgs.clamp(0, 255)

    def results_as_numpy(self) -> np.ndarray:
        """Return the batch results as a numpy array.

        Returns:
            (B, H, W, 3) uint8 numpy array.
        """
        return self.resulting_images.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
