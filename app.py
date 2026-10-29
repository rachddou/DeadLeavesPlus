"""Vibrant Leaves Generator: interactive app to generate a sample of 9 dead leaves images from parameters
chosen in a menu, then a full dataset of 1000 images with the same parameters.

The page (app_ui/index.html) is served by a small HTTP server; the images are
generated in parallel worker processes with the same code as image_generation.py.
Sample images are streamed back to the page as they are ready; dataset images
are saved in a new folder of config io.path_origin (datasets/ by default),
together with the config used.

usage:
    python app.py [--host 127.0.0.1] [--port 7860] [--workers N]
"""
import argparse
import base64
import concurrent.futures as cf
import io
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parent
UI_PAGE = ROOT / "app_ui" / "index.html"
DEFAULTS = OmegaConf.load(ROOT / "config" / "default.yaml")
N_IMAGES = 9
N_DATASET = 1000
DATASET_ROOT = ROOT / DEFAULTS.io.path_origin
SHAPE_TYPES = ["poly", "disk", "rectangle", "mix"]
TEXTURE_TYPES = ["sin", "freq_noise", "texture_mixes"]
IMAGE_SIZES = [128, 256, 384, 512, 768, 1024]
ICON_SIZE = 128


# ---------------------------------------------------------------------------
# Worker processes
# ---------------------------------------------------------------------------

def _generate(config, seed):
    """Generates one image (runs in a worker process)."""
    from dead_leaves_generation.utils.utils import set_seed
    from image_generation import build_generator, generate_image

    config = OmegaConf.create(config)
    generator = build_generator(config)
    set_seed(seed)
    return generate_image(generator, config)


def _generate_to_file(config, seed, path):
    """Generates one image and saves it (runs in a worker process)."""
    Image.fromarray(_generate(config, seed)).save(path)


def _texture_icon(texture_type, image_dir):
    """A sample texture of the given type (runs in a worker process)."""
    from dead_leaves_generation.dead_leaves_class import Textures
    from dead_leaves_generation.utils.utils import set_seed

    set_seed(TEXTURE_TYPES.index(texture_type) + 3)
    textures = Textures(width=ICON_SIZE, use_natural_images=Path(image_dir).is_dir(),
                        path=image_dir, types=[texture_type])
    return textures.generate_texture(width=ICON_SIZE)[:ICON_SIZE, :ICON_SIZE]


def _warm_up():
    """Compiles / loads the numba kernels once per worker, so that the first
    sample is not slowed down by it."""
    config = _base_config()
    config.update(image_size=128)
    config["color"]["use_natural_images"] = False
    config["texture"].update(types=TEXTURE_TYPES,
                             type_weights=[1 / len(TEXTURE_TYPES)] * len(TEXTURE_TYPES))
    try:
        for seed in range(3):
            _generate(config, seed)
    except Exception:  # the real error, if any, shows up when generating
        pass


# ---------------------------------------------------------------------------
# Menu <-> config
# ---------------------------------------------------------------------------

def _base_config():
    config = OmegaConf.to_container(DEFAULTS)
    config.pop("defaults", None)
    config["image_type"] = "dead_leaves"
    return config


def _default_image_dir():
    configured = Path(DEFAULTS.color.image_dir)
    fallback = ROOT / "datasets" / "test_sets" / "Kodak24"
    if not configured.is_dir() and fallback.is_dir():
        return str(fallback)
    return str(configured)


def menu_defaults():
    """Initial state of the menu, from config/default.yaml."""
    d = DEFAULTS
    weights = np.array(d.texture.type_weights, dtype=float)
    return {
        "shape_type": d.shape.shape_type,
        "radius_min": d.shape.radius_min,
        "radius_max": d.shape.radius_max,
        "power_law_exponent": d.shape.power_law_exponent,
        "texture_enabled": d.texture.enabled,
        "texture_shares": dict(zip(d.texture.types, (weights / weights.sum()).tolist())),
        "slope_min": d.texture.slope_range[0][0],
        "slope_max": d.texture.slope_range[0][1],
        "apply_warp": d.texture.apply_warp,
        "apply_perspective": d.texture.apply_perspective,
        "use_natural_images": d.color.use_natural_images,
        "image_dir": _default_image_dir(),
        "grey": d.color.grey,
        "depth_of_field": d.post_process.depth_of_field,
        "extra_blur": d.post_process.extra_blur,
        "downscale": d.post_process.downscale,
        "image_size": d.image_size,
    }


def build_config(params):
    """Turns the menu state sent by the page into a config dict.

    Raises:
        ValueError: on invalid parameters (the message is shown on the page).
    """
    config = _base_config()
    shape, texture, color, post = (config["shape"], config["texture"],
                                   config["color"], config["post_process"])

    if params["shape_type"] not in SHAPE_TYPES:
        raise ValueError(f"Unknown shape type {params['shape_type']!r}.")
    shape["multiple_shapes"] = False
    shape["shape_type"] = params["shape_type"]
    shape["radius_min"] = int(params["radius_min"])
    shape["radius_max"] = int(params["radius_max"])
    if not 1 <= shape["radius_min"] < shape["radius_max"]:
        raise ValueError("The minimum radius must be at least 1 and smaller than the maximum.")
    shape["power_law_exponent"] = float(params["power_law_exponent"])
    if shape["power_law_exponent"] <= 1:
        raise ValueError("The power-law exponent must be larger than 1.")

    texture["enabled"] = bool(params["texture_enabled"])
    shares = {t: float(params["texture_shares"].get(t, 0)) for t in TEXTURE_TYPES}
    shares = {t: s for t, s in shares.items() if s > 0}
    if texture["enabled"]:
        if not shares:
            raise ValueError("Select at least one texture type.")
        total = sum(shares.values())
        texture["types"] = list(shares)
        texture["type_weights"] = [s / total for s in shares.values()]
    slope_min, slope_max = float(params["slope_min"]), float(params["slope_max"])
    if not 0 <= slope_min < slope_max:
        raise ValueError("The 1/f slope range must satisfy 0 <= min < max.")
    texture["slope_range"] = [[slope_min, slope_max]]
    texture["apply_warp"] = bool(params["apply_warp"])
    texture["apply_perspective"] = bool(params["apply_perspective"])

    color["use_natural_images"] = bool(params["use_natural_images"])
    color["image_dir"] = str(params["image_dir"])
    if color["use_natural_images"] and not Path(color["image_dir"]).is_dir():
        raise ValueError(f"Colour image directory not found: {color['image_dir']}")
    color["grey"] = bool(params["grey"])

    post["depth_of_field"] = bool(params["depth_of_field"])
    # Gaussian depth of field; without it the layers are not blurred at all
    post["blur_type"] = "gaussian" if post["depth_of_field"] else "None"
    post["extra_blur"] = bool(params["extra_blur"])
    post["downscale"] = bool(params["downscale"])

    config["image_size"] = int(params["image_size"])
    if config["image_size"] not in IMAGE_SIZES:
        raise ValueError(f"Image size must be one of {IMAGE_SIZES}.")
    return config


# ---------------------------------------------------------------------------
# Icons
# ---------------------------------------------------------------------------

def _display_path(path):
    """Path relative to the current directory when it is inside it."""
    relative = os.path.relpath(path)
    return str(Path(path).resolve()) if relative.startswith("..") else relative


def _png(image):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _paste(canvas, mask, center, size):
    """Draws a binary mask, scaled to fit in size x size, centred on center."""
    h, w = mask.shape
    scale = size / max(h, w)
    small = cv2.resize(np.uint8(mask) * 255, (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_AREA)
    r0, c0 = center[0] - small.shape[0] // 2, center[1] - small.shape[1] // 2
    region = canvas[r0:r0 + small.shape[0], c0:c0 + small.shape[1]]
    np.maximum(region, small, out=region)


def shape_icons():
    """Silhouettes (alpha masks) of each shape type, drawn by the generator."""
    from dead_leaves_generation.dead_leaves_class import disk_mask
    from dead_leaves_generation.polygons_maker_bis import binary_polygon_generator, make_rectangle_mask
    from dead_leaves_generation.utils.utils import set_seed

    set_seed(11)
    poly = binary_polygon_generator(121, n=90, allow_holes=False, smoothing=True)
    disk = disk_mask(40)
    rect = make_rectangle_mask(40)
    s, c = ICON_SIZE, ICON_SIZE // 2
    layouts = {
        "poly": [(poly, (c, c), 0.85 * s)],
        "disk": [(disk, (c, c), 0.7 * s)],
        "rectangle": [(rect, (c, c), 0.8 * s)],
        "mix": [(poly, (int(0.32 * s), int(0.32 * s)), 0.5 * s),
                (disk, (int(0.72 * s), int(0.3 * s)), 0.36 * s),
                (rect, (int(0.62 * s), int(0.7 * s)), 0.48 * s)],
    }
    icons = {}
    for name, parts in layouts.items():
        alpha = np.zeros((s, s), np.uint8)
        for mask, center, size in parts:
            _paste(alpha, mask, center, int(size))
        icons[name] = _png(np.dstack([np.full((s, s, 3), 255, np.uint8), alpha]))
    return icons


def texture_icons(pool, image_dir):
    futures = {t: pool.submit(_texture_icon, t, image_dir) for t in TEXTURE_TYPES}
    return {t: _png(f.result()) for t, f in futures.items()}


# ---------------------------------------------------------------------------
# Full dataset generation
# ---------------------------------------------------------------------------

class DatasetJob:
    """Generates a dataset in the background, keeping at most max_in_flight
    images queued so that sample requests are not stuck behind the dataset."""

    def __init__(self, pool, config, n_images, max_in_flight, root):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = Path(root) / f"dataset_{stamp}"
        self.out_dir.mkdir(parents=True, exist_ok=False)
        # saved config: image_generation.py --config-path <out_dir> --config-name config
        # generates a dataset with the same parameters
        saved = dict(config, n_images=n_images,
                     io={"path_origin": str(self.out_dir.parent), "path": self.out_dir.name})
        OmegaConf.save(OmegaConf.create(saved), self.out_dir / "config.yaml")
        self.pool, self.config = pool, config
        self.total, self.done, self.errors, self.last_error = n_images, 0, 0, ""
        self.max_in_flight = max_in_flight
        self.cancelled = False
        self.state = "running"
        self.started, self.finished = time.time(), None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        seeds = random.sample(range(2 ** 31), self.total)
        in_flight, next_index = set(), 0
        while (next_index < self.total or in_flight) and not self.cancelled:
            while next_index < self.total and len(in_flight) < self.max_in_flight:
                path = self.out_dir / f"im_{next_index:04d}.png"
                in_flight.add(self.pool.submit(_generate_to_file, self.config,
                                               seeds[next_index], str(path)))
                next_index += 1
            finished, in_flight = cf.wait(in_flight, return_when=cf.FIRST_COMPLETED)
            self._count(finished)
        # cancelled: drop the queued images, the ones being generated are still saved
        for future in in_flight:
            future.cancel()
        self._count(cf.wait(in_flight).done)
        self.state = "cancelled" if self.cancelled else "done"
        self.finished = time.time()

    def _count(self, futures):
        for future in futures:
            if future.cancelled():
                continue
            try:
                future.result()
                self.done += 1
            except Exception as error:
                self.errors += 1
                self.last_error = f"{type(error).__name__}: {error}"

    def cancel(self):
        self.cancelled = True

    def status(self):
        end = self.finished or time.time()
        return {"state": self.state, "done": self.done, "errors": self.errors,
                "last_error": self.last_error, "total": self.total,
                "elapsed": end - self.started, "out_dir": str(self.out_dir)}


def open_folder(path):
    """Opens a folder in the file manager of the machine running the app.

    Raises:
        OSError / subprocess.SubprocessError: when it cannot be opened (e.g. on
        a remote server without a desktop).
    """
    if os.name == "nt":
        os.startfile(path)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.run([opener, str(path)], check=True, timeout=10,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    pool = None
    workers = 1
    page_state = None
    dataset_job = None
    dataset_root = DATASET_ROOT
    dataset_lock = threading.Lock()

    def log_message(self, format, *args):
        pass

    def _send(self, body, content_type, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data, status=HTTPStatus.OK):
        self._send(json.dumps(data).encode(), "application/json", status)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(UI_PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._send_json(self.page_state)
        elif self.path == "/api/dataset":
            job = Handler.dataset_job
            self._send_json(job.status() if job else {"state": "idle"})
        else:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _read_config(self):
        """Config from the menu state in the request body, or None after
        answering with an error."""
        try:
            params = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            return build_config(params)
        except (ValueError, KeyError, TypeError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return None

    def do_POST(self):
        if self.path == "/api/generate":
            self._generate_sample()
        elif self.path == "/api/dataset":
            self._start_dataset()
        elif self.path == "/api/dataset/cancel":
            job = Handler.dataset_job
            if job:
                job.cancel()
            self._send_json(job.status() if job else {"state": "idle"})
        elif self.path == "/api/dataset/open":
            # only the folder of the current dataset can be opened
            job = Handler.dataset_job
            if not job:
                return self._send_json({"error": "No dataset yet."}, HTTPStatus.NOT_FOUND)
            try:
                open_folder(job.out_dir)
            except (OSError, subprocess.SubprocessError):
                return self._send_json({"error": "Cannot open the folder on this machine."},
                                       HTTPStatus.INTERNAL_SERVER_ERROR)
            self._send_json({"opened": str(job.out_dir)})
        else:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _start_dataset(self):
        config = self._read_config()
        if config is None:
            return
        with Handler.dataset_lock:
            job = Handler.dataset_job
            if job and job.state == "running":
                return self._send_json({"error": "A dataset is already being generated."},
                                       HTTPStatus.CONFLICT)
            try:
                Handler.dataset_job = DatasetJob(self.pool, config, N_DATASET,
                                                 max_in_flight=self.workers,
                                                 root=self.dataset_root)
            except OSError as error:
                return self._send_json({"error": f"Cannot create the dataset folder: {error}"},
                                       HTTPStatus.INTERNAL_SERVER_ERROR)
        self._send_json(Handler.dataset_job.status())

    def _generate_sample(self):
        config = self._read_config()
        if config is None:
            return

        # one JSON line per image, in completion order
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seeds = random.sample(range(2 ** 31), N_IMAGES)
        futures = {self.pool.submit(_generate, config, seed): i for i, seed in enumerate(seeds)}
        try:
            for future in cf.as_completed(futures):
                line = {"index": futures[future]}
                try:
                    line["image"] = _png(future.result())
                except Exception as error:
                    line["error"] = f"{type(error).__name__}: {error}"
                self.wfile.write((json.dumps(line) + "\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # the page started a new sample: drop the images still queued
            for future in futures:
                future.cancel()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1",
                        help="use 0.0.0.0 to access the app from another machine")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--workers", type=int, default=min(N_IMAGES, os.cpu_count() or 1),
                        help="parallel generation processes")
    parser.add_argument("--dataset-dir", default=str(DATASET_ROOT),
                        help="where full datasets are saved (default: config io.path_origin)")
    args = parser.parse_args()

    with cf.ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"),
                                initializer=_warm_up) as pool:
        defaults = menu_defaults()
        Handler.pool = pool
        Handler.workers = args.workers
        Handler.dataset_root = Path(args.dataset_dir)
        Handler.page_state = {
            "defaults": defaults,
            "n_images": N_IMAGES,
            "n_dataset": N_DATASET,
            "dataset_root": _display_path(args.dataset_dir),
            "shape_types": SHAPE_TYPES,
            "texture_types": TEXTURE_TYPES,
            "image_sizes": IMAGE_SIZES,
            "icons": {"shapes": shape_icons(),
                      "textures": texture_icons(pool, defaults["image_dir"])},
        }
        server = ThreadingHTTPServer((args.host, args.port), Handler)
        server.daemon_threads = True
        print(f"Vibrant Leaves Generator running on http://{args.host}:{args.port} (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
