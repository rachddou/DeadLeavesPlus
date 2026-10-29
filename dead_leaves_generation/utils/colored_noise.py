import numpy as np
from scipy import fft as sfft

from dead_leaves_generation.utils.kernels import gather_planes, impose_magnitude, planes_to_uint8

# True: synthesise the noise on exactly the requested width x width grid, like
# the original implementation (e.g. to compare images with it). False (faster):
# synthesise it on the next FFT-friendly size (2^a 3^b 5^c, median +2%) and crop;
# FFTs of awkward sizes (large prime factors) are several times slower.
EXACT_ORIGINAL = False


def _magnitude_1d(resolution, slope):
    """|f / resolution|^slope for the FFT frequency index f of each array index.

    The imposed magnitude is M[i, j] = 1 / (1e-16 + g[i] + g[j]) with
    g = _magnitude_1d(width, slope), and M[0, 0] = 0 (no DC component).
    """
    k = np.arange(resolution)
    f = ((k - resolution // 2) % resolution) - resolution / 2
    return np.abs(f / resolution) ** slope


def sample_color_noise(image,width,slope,region=None):
    """function that creates colored noise with a given slope in the frequency domain from the color histogram of an existing image

    Each channel of a white noise image (pixels drawn from a 50x50 patch of
    ``image``) keeps its Fourier phase, gets the 1/f^slope magnitude M (see
    _magnitude_1d), and is rescaled to the mean / std of the white noise. The
    noise is synthesised on the next FFT-friendly size and cropped to width x
    width (see EXACT_ORIGINAL).

    Args:
        image (_type_): source image for color histogram
        width (_type_): size of the texture image
        slope (_type_): slope of the magnitude map in the log domain
        region (tuple, optional): (r0, r1, c0, c1) part of the texture that is
            needed; the rest of the returned texture is left black. The noise is
            still synthesised on the whole grid. Defaults to None.
    """
    n = width if EXACT_ORIGINAL else sfft.next_fast_len(width, real=True)
    h,w = image.shape[0],image.shape[1]
    p = 50

    xx,yy = np.random.randint(0,h-p),np.random.randint(0,w-p)
    image = image[xx:xx+p,yy:yy+p]
    image = image.reshape((p*p,3))
    # same random stream as np.random.choice(np.arange(p*p), replace=True, size=...)
    index = np.random.randint(0, p*p, size=n*n)

    planes = gather_planes(np.ascontiguousarray(image), index, n)
    m = planes.mean(axis=(1, 2), dtype=np.float32)
    s = planes.std(axis=(1, 2), dtype=np.float32)

    # real(ifft2(magnitude * exp(i * angle(fft2(x - m))))) computed with real FFTs
    spectrum = sfft.rfft2(planes - m[:, None, None], axes=(1, 2))
    g = _magnitude_1d(n, slope)
    impose_magnitude(spectrum, g, g[(-np.arange(n)) % n])
    y = sfft.irfft2(spectrum, s=(n, n), axes=(1, 2), overwrite_x=True)
    if n != width:
        # the texture is normalised on what is kept, as without the crop
        y = y[:, :width, :width]

    mean, std = y.mean(axis=(1, 2)), y.std(axis=(1, 2))
    if region is None:
        return planes_to_uint8(y, mean, std, s, m)
    r0, r1, c0, c1 = region
    image = np.zeros((width, width, 3), dtype=np.uint8)
    image[r0:r1, c0:c1] = planes_to_uint8(y[:, r0:r1, c0:c1], mean, std, s, m)
    return image
