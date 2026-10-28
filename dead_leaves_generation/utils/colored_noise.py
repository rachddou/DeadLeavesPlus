import numpy as np
from scipy import fft as sfft

from dead_leaves_generation.utils.kernels import gather_planes, impose_magnitude, planes_to_uint8


def _magnitude_1d(resolution, slope):
    """|f|^slope along one axis of the fftshift-ed frequency grid of sample_magnitude."""
    k = np.arange(resolution)
    f = ((k - resolution // 2) % resolution) - resolution / 2
    return np.abs(f / resolution) ** slope


def sample_magnitude(resolution,slope):
    """function that creates a magnitude map for the frequency domain in 1/f^slope

    Args:
        resolution (int): size of the texture image
        slope (float): slope of the magnitude map in the log domain
    """
    g = _magnitude_1d(resolution, slope)
    magnitude = 1 / (1e-16 + g[None, :] + g[:, None])
    magnitude[0,0] = 0
    magnitude = np.tile(magnitude[:,:,None], (1,1,3))
    return(magnitude)


def sample_color_noise(image,width,slope):
    """function that creates colored noise with a given slope in the frequency domain from the color histogram of an existing image

    Args:
        image (_type_): source image for color histogram
        width (_type_): size of the texture image
        slope (_type_): slope of the magnitude map in the log domain
    """
    h,w = image.shape[0],image.shape[1]
    p = 50

    xx,yy = np.random.randint(0,h-p),np.random.randint(0,w-p)
    image = image[xx:xx+p,yy:yy+p]
    image = image.reshape((p*p,3))
    # same random stream as np.random.choice(np.arange(p*p), replace=True, size=...)
    index = np.random.randint(0, p*p, size=width*width)

    planes = gather_planes(np.ascontiguousarray(image), index, width)
    m = planes.mean(axis=(1, 2), dtype=np.float32)
    s = planes.std(axis=(1, 2), dtype=np.float32)

    # real(ifft2(magnitude * exp(i * angle(fft2(x - m))))) computed with real FFTs
    spectrum = sfft.rfft2(planes - m[:, None, None], axes=(1, 2))
    g = _magnitude_1d(width, slope)
    impose_magnitude(spectrum, g, g[(-np.arange(width)) % width])
    y = sfft.irfft2(spectrum, s=(width, width), axes=(1, 2), overwrite_x=True)

    return planes_to_uint8(y, y.mean(axis=(1, 2)), y.std(axis=(1, 2)), s, m)
