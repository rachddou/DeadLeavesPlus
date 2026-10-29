"""Numba kernels for the hot loops of the generator.

Each kernel fuses a chain of numpy operations of the reference implementation
into a single pass, using the same formulas and constants so that results match
the numpy versions up to floating point rounding.
"""
import math

import numpy as np
from numba import njit
from skimage.color.colorconv import rgb_from_xyz, xyz_from_rgb

# D65 / 2° reference white, as used by skimage.color.rgb2lab / lab2rgb
_WHITE = np.array([0.95047, 1.0, 1.08883])
_RGB_FROM_XYZ = np.ascontiguousarray(rgb_from_xyz, dtype=np.float64)
_XYZ_FROM_RGB = np.ascontiguousarray(xyz_from_rgb, dtype=np.float64)


# ---------------------------------------------------------------------------
# Colour conversions (same formulas as skimage.color.lab2rgb / rgb2lab)
# ---------------------------------------------------------------------------

@njit(cache=True, inline="always")
def _lab_to_rgb_px(L, a, b, M, out):
    y = (L + 16.0) / 116.0
    x = (a / 500.0) + y
    z = y - (b / 200.0)
    if z < 0:
        z = 0.0
    if x > 0.2068966:
        x = x ** 3.0
    else:
        x = (x - 16.0 / 116.0) / 7.787
    if y > 0.2068966:
        y = y ** 3.0
    else:
        y = (y - 16.0 / 116.0) / 7.787
    if z > 0.2068966:
        z = z ** 3.0
    else:
        z = (z - 16.0 / 116.0) / 7.787
    x *= 0.95047
    z *= 1.08883
    for c in range(3):
        v = x * M[c, 0] + y * M[c, 1] + z * M[c, 2]
        if v > 0.0031308:
            v = 1.055 * v ** (1 / 2.4) - 0.055
        else:
            v *= 12.92
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        out[c] = v


@njit(cache=True)
def _mix_lab_to_rgb_u8(interp, t1, t2, M):
    """uint8(255 * lab2rgb(interp * (t2 - t1) + t1)), with t1, t2 broadcastable
    (H, W, 3) Lab images (strides may be 0 for constant colours)."""
    h, w = interp.shape
    out = np.empty((h, w, 3), np.uint8)
    px = np.empty(3)
    for i in range(h):
        for j in range(w):
            m = interp[i, j]
            L = m * (t2[i, j, 0] - t1[i, j, 0]) + t1[i, j, 0]
            a = m * (t2[i, j, 1] - t1[i, j, 1]) + t1[i, j, 1]
            b = m * (t2[i, j, 2] - t1[i, j, 2]) + t1[i, j, 2]
            _lab_to_rgb_px(L, a, b, M, px)
            for c in range(3):
                out[i, j, c] = np.uint8(px[c] * 255)
    return out


def mix_lab_to_rgb_u8(interp, t1, t2):
    """Fused ``np.uint8(lab2rgb(interp * (t2 - t1) + t1) * 255)``.

    Args:
        interp (np.ndarray): (H, W) interpolation map.
        t1, t2 (np.ndarray): Lab colours, either (3,), (1, 1, 3) or (H, W, 3).
    """
    interp = np.asarray(interp, dtype=np.float64)
    shape = interp.shape + (3,)
    return _mix_lab_to_rgb_u8(interp, _as_lab_image(t1, shape),
                              _as_lab_image(t2, shape), _RGB_FROM_XYZ)


def _as_lab_image(t, shape):
    t = np.asarray(t, dtype=np.float64)
    if t.size == 3:
        t = t.reshape(3)
    return np.broadcast_to(t, shape)


@njit(cache=True, inline="always")
def _lin_rgb_to_lab_px(r, g, b, M, white, out, i):
    x = (r * M[0, 0] + g * M[0, 1] + b * M[0, 2]) / white[0]
    y = (r * M[1, 0] + g * M[1, 1] + b * M[1, 2]) / white[1]
    z = (r * M[2, 0] + g * M[2, 1] + b * M[2, 2]) / white[2]
    if x > 0.008856:
        x = np.cbrt(x)
    else:
        x = 7.787 * x + 16.0 / 116.0
    if y > 0.008856:
        y = np.cbrt(y)
    else:
        y = 7.787 * y + 16.0 / 116.0
    if z > 0.008856:
        z = np.cbrt(z)
    else:
        z = 7.787 * z + 16.0 / 116.0
    out[i, 0] = (116.0 * y) - 16.0
    out[i, 1] = 500.0 * (x - y)
    out[i, 2] = 200.0 * (y - z)


@njit(cache=True, inline="always")
def _srgb_to_linear(v):
    if v > 0.04045:
        return ((v + 0.055) / 1.055) ** 2.4
    return v / 12.92


@njit(cache=True)
def _rgb_to_lab(rgb, div, M, white):
    n = rgb.shape[0]
    out = np.empty((n, 3))
    for i in range(n):
        _lin_rgb_to_lab_px(_srgb_to_linear(rgb[i, 0] / div), _srgb_to_linear(rgb[i, 1] / div),
                           _srgb_to_linear(rgb[i, 2] / div), M, white, out, i)
    return out


@njit(cache=True)
def _rgb_u8_to_lab(rgb, lut, M, white):
    n = rgb.shape[0]
    out = np.empty((n, 3))
    for i in range(n):
        _lin_rgb_to_lab_px(lut[rgb[i, 0]], lut[rgb[i, 1]], lut[rgb[i, 2]], M, white, out, i)
    return out


@njit(cache=True)
def _srgb_lut():
    lut = np.empty(256)
    for v in range(256):
        lut[v] = _srgb_to_linear(v / 255.0)
    return lut


def rgb_to_lab(rgb, div=255.0):
    """Equivalent of ``rgb2lab(rgb / div)`` for an (..., 3) array."""
    rgb = np.asarray(rgb)
    shape = rgb.shape
    if rgb.dtype == np.uint8 and div == 255.0:
        # only 256 possible values per channel: linearise through a table
        flat = np.ascontiguousarray(rgb.reshape(-1, 3))
        return _rgb_u8_to_lab(flat, _srgb_lut(), _XYZ_FROM_RGB, _WHITE).reshape(shape)
    flat = np.ascontiguousarray(rgb.reshape(-1, 3), dtype=np.float64)
    return _rgb_to_lab(flat, div, _XYZ_FROM_RGB, _WHITE).reshape(shape)


# ---------------------------------------------------------------------------
# Interpolation map helpers
# ---------------------------------------------------------------------------

@njit(cache=True)
def sinusoid_transfer(x, thresh, lam, out):
    """Maps a sinusoid in [-1, 1] to a sharpened profile in [0, 1]:

        x = 0.5 + 0.5 * x ; x[x > thresh] = 1 ; x = sigmoid(x)
        x = (x - sigmoid(0)) / (sigmoid(1) - sigmoid(0))

    with sigmoid(x) = 1 / (1 + exp(-lam * x)). ``out`` may alias ``x``.
    """
    s0 = 1 / (1 + math.exp(-lam * 0.0))
    s1 = 1 / (1 + math.exp(-lam * 1.0))
    d = s1 - s0
    h, w = x.shape
    for i in range(h):
        for j in range(w):
            v = 0.5 + 0.5 * x[i, j]
            if v > thresh:
                v = 1.0
            v = 1 / (1 + math.exp(-lam * v))
            out[i, j] = (v - s0) / d


@njit(cache=True)
def normalized_logistic_inplace(x, k):
    """In place: x <- (logistic(12.1 x - 6) - logistic(-6)) / (logistic(6.1) - logistic(-6))
    with logistic(t) = 1 / (1 + exp(-k t)), a smooth step from 0 to 1 on [0, 1]."""
    lo = 1 / (1 + math.exp(-k * (-6 - 0)))
    hi = 1 / (1 + math.exp(-k * (6.1 - 0)))
    d = hi - lo
    h, w = x.shape
    for i in range(h):
        for j in range(w):
            v = 1 / (1 + math.exp(-k * ((12.1 * x[i, j] - 6) - 0)))
            x[i, j] = (v - lo) / d


@njit(cache=True)
def warp_bilinear_clipped(im, u, v):
    """out[i, j] = bilinear interpolation of im at (row i + v[i, j], col j + u[i, j]),
    plus im.min(). The corner indices are clipped to the image before computing
    the weights, so samples outside [0, w-1) x [0, h-1) get cancelling weights
    and evaluate to im.min(), as in the original implementation."""
    h, w = im.shape
    out = np.empty((h, w))
    mn = im[0, 0]
    for i in range(h):
        for j in range(w):
            if im[i, j] < mn:
                mn = im[i, j]
    for i in range(h):
        for j in range(w):
            xx = u[i, j] + j
            yy = v[i, j] + i
            x0 = int(math.floor(xx))
            y0 = int(math.floor(yy))
            x1 = x0 + 1
            y1 = y0 + 1
            x0 = min(max(x0, 0), w - 1)
            x1 = min(max(x1, 0), w - 1)
            y0 = min(max(y0, 0), h - 1)
            y1 = min(max(y1, 0), h - 1)
            wa = (x1 - xx) * (y1 - yy)
            wb = (x1 - xx) * (yy - y0)
            wc = (xx - x0) * (y1 - yy)
            wd = (xx - x0) * (yy - y0)
            out[i, j] = (wa * im[y0, x0] + wb * im[y1, x0]
                         + wc * im[y0, x1] + wd * im[y1, x1]) + mn
    return out


# ---------------------------------------------------------------------------
# Polygon rasterisation (GDAL / rasterio.features.rasterize rule)
# ---------------------------------------------------------------------------

@njit(cache=True)
def fill_polygon(xs, ys, ring_offsets, height, width):
    """Rasterise polygon rings with the even-odd rule used by GDAL's
    ``GDALdllImageFilledPolygon`` (pixel-centre sampling, identity transform).

    Args:
        xs, ys: concatenated ring coordinates (x = column, y = row).
        ring_offsets: start index of each ring, plus the total length.
    """
    out = np.zeros((height, width), np.bool_)
    n = xs.shape[0]
    if n == 0:
        return out
    ints = np.empty(n)
    ymin = ys.min()
    ymax = ys.max()
    r0 = max(0, int(ymin))
    r1 = min(height - 1, int(ymax))
    n_rings = ring_offsets.shape[0] - 1
    for row in range(r0, r1 + 1):
        dy = row + 0.5
        k = 0
        for r in range(n_rings):
            s = ring_offsets[r]
            e = ring_offsets[r + 1]
            for i in range(s, e):
                if i == s:
                    i1 = e - 1
                    i2 = s
                else:
                    i1 = i - 1
                    i2 = i
                dy1 = ys[i1]
                dy2 = ys[i2]
                if (dy1 < dy and dy2 < dy) or (dy1 > dy and dy2 > dy):
                    continue
                if dy1 < dy2:
                    dx1 = xs[i1]
                    dx2 = xs[i2]
                elif dy1 > dy2:
                    dy2 = ys[i1]
                    dy1 = ys[i2]
                    dx2 = xs[i1]
                    dx1 = xs[i2]
                else:
                    # horizontal edge lying exactly on a pixel-centre line
                    if xs[i1] > xs[i2]:
                        a = max(int(math.floor(xs[i2] + 0.5)), 0)
                        b = min(int(math.floor(xs[i1] + 0.5)), width)
                        for c in range(a, b):
                            out[row, c] = True
                    continue
                if dy < dy2 and dy >= dy1:
                    ints[k] = (dy - dy1) * (dx2 - dx1) / (dy2 - dy1) + dx1
                    k += 1
        if k < 2:
            continue
        seg = np.sort(ints[:k])
        for i in range(0, k - 1, 2):
            a = int(math.floor(seg[i] + 0.5))
            b = int(math.floor(seg[i + 1] + 0.5))
            if a < 0:
                a = 0
            if b > width:
                b = width
            for c in range(a, b):
                out[row, c] = True
    return out


# ---------------------------------------------------------------------------
# Coloured noise (colored_noise.sample_color_noise)
# ---------------------------------------------------------------------------

@njit(cache=True)
def gather_planes(colors, index, width):
    """(3, width, width) float32 planes of colors[index] (colors: (N, 3) uint8)."""
    out = np.empty((3, width, width), np.float32)
    for i in range(width):
        for j in range(width):
            k = index[i * width + j]
            for c in range(3):
                out[c, i, j] = colors[k, c]
    return out


@njit(cache=True)
def impose_magnitude(spectrum, g, g_neg):
    """In place: spectrum <- (M(k) + M(-k)) / 2 * exp(i angle(spectrum)) on the
    rfft half-plane, M[i, j] = 1 / (1e-16 + g[i] + g[j]) being the magnitude
    built from the 1D profile g (colored_noise._magnitude_1d), and g_neg = g at
    negated frequencies."""
    nc, h, w = spectrum.shape
    for i in range(h):
        for j in range(w):
            if i == 0 and j == 0:
                mag = 0.0
            else:
                mag = 0.5 * (1 / (1e-16 + g[i] + g[j]) + 1 / (1e-16 + g_neg[i] + g_neg[j]))
            for c in range(nc):
                z = spectrum[c, i, j]
                a = abs(z)
                if a > 0:
                    spectrum[c, i, j] = z * (mag / a)
                else:
                    spectrum[c, i, j] = mag


@njit(cache=True)
def planes_to_uint8(y, mean, std, s, m):
    """Interleaved uint8(clip(float32((y - mean) / std * s + m), 0, 255))."""
    nc, h, w = y.shape
    out = np.empty((h, w, nc), np.uint8)
    for c in range(nc):
        for i in range(h):
            for j in range(w):
                v = np.float32((y[c, i, j] - mean[c]) / std[c] * s[c] + m[c])
                if v < 0:
                    v = 0
                elif v > 255:
                    v = 255
                out[i, j, c] = np.uint8(v)
    return out


# ---------------------------------------------------------------------------
# Polygon smoothing
# ---------------------------------------------------------------------------

@njit(cache=True)
def _correlate_nearest_sym(src, fw, dst):
    """scipy.ndimage.correlate1d along axis 0 with a symmetric kernel and
    mode='nearest' (same per-element summation order as scipy's NI_Correlate1D)."""
    h, w = src.shape
    r = fw.shape[0] // 2
    acc = np.empty(w)
    for i in range(h):
        for j in range(w):
            acc[j] = src[i, j] * fw[r]
        for k in range(r, 0, -1):
            a = src[max(i - k, 0)]
            b = src[min(i + k, h - 1)]
            f = fw[r - k]
            for j in range(w):
                acc[j] += (np.float64(a[j]) + np.float64(b[j])) * f
        for j in range(w):
            dst[i, j] = acc[j]


@njit(cache=True)
def gaussian_threshold(mask, fw, thresh):
    """``gaussian_filter(mask.astype(np.float32), sigma, mode='nearest') > thresh``
    for a symmetric 1D kernel fw (float32 intermediate, as in scipy)."""
    h, w = mask.shape
    src = np.empty((h, w), np.float32)
    for i in range(h):
        for j in range(w):
            src[i, j] = mask[i, j]
    tmp = np.empty((h, w), np.float32)
    _correlate_nearest_sym(src, fw, tmp)
    tmp_t = np.ascontiguousarray(tmp.T)
    out_t = np.empty((w, h), np.float32)
    _correlate_nearest_sym(tmp_t, fw, out_t)
    out = np.empty((h, w), np.bool_)
    for i in range(h):
        for j in range(w):
            out[i, j] = out_t[j, i] > thresh
    return out


@njit(cache=True)
def seed_numba(seed):
    """Seeds the random generator used inside numba-compiled functions, which
    is separate from numpy's global generator."""
    np.random.seed(seed)
