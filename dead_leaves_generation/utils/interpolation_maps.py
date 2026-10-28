import numpy as np
from dead_leaves_generation.utils.utils import (rotate_cv, normalize,
                                                rotation_inverse, source_window, warp_window)
from dead_leaves_generation.utils.geometric_perturbation import generate_perturbation
from dead_leaves_generation.utils.kernels import sinusoid_transfer
from numba import njit
TMIN = 5
TMAX = 100
def sample_grid(width = 100,period = [100],angle = 45):
    """function that creates a grid pattern with a given orientation

    The grid is drawn on a 2*width canvas, its central width x width crop is
    normalised and rotated (with black corners).

    Args:
        width (int, optional): width of the final image. Defaults to 100.
        period (list, optional): period of the grid. If len(period) is 2, then the grid is bi-directional. Defaults to [100].
        angle (int, optional): orientation of the grid. Defaults to 45.
    """
    thickness =  np.random.randint(1,3)
    x = np.ones((2*width))
    for i in range(1,1+(2*width)//(period[0]+thickness)):
        x[i*(period[0]+thickness)-thickness:i*(period[0]+thickness)] = 0
    # central width x width crop of the 2*width canvas
    crop = slice(width//2, width//2 + width)
    grid = np.broadcast_to(x[crop], (width, width))

    if len(period)==2:
        y =  np.ones((2*width))
        for i in range(1,1+(2*width)//(period[1]+thickness)):
            y[i*(period[1]+thickness)-thickness:i*(period[1]+thickness)] = 0
        grid  = grid*y[crop, None]
    grid = normalize(grid)
    grid = rotate_cv(grid,angle)
    return(grid)

@njit(cache=True)
def sample_period(n_period):
    periods = np.floor(TMIN + (TMAX-TMIN)*np.random.random(size=n_period))
    return(periods)

def _sample_1d_sinusoid(n):
    period = sample_period(1)
    return np.sin(((2*np.pi)/period[0])*np.arange(0,n))

def sample_sinusoid(width = 100,angle = 45 ,angle1 = 45,angle2 = 45):
    """function that create a pseudo-periodic grey-level pattern based on sinusoidal functions.
    This patterns then serves as an interpolation map either between two colors or two texture maps.

    A sinusoid (random period) is rotated by ``angle`` on a (2*width)^2 canvas,
    saturated above a random threshold and sharpened by a sigmoid. Half of the
    time it is multiplied by a second one along ``angle1``. The product is
    rotated by ``angle2`` and its central width x width crop is normalised.
    Only the windows of the intermediate images that contribute to that crop are
    computed (see utils.warp_window).

    Args:
        width (int, optional): width of the final image. Defaults to 100.
        angle (int, optional): angle for dimension 1. Defaults to 45.
        angle1 (int, optional): angle for dimension 2. Defaults to 45.
        angle2 (int, optional): rotation applied to the whole sinusoidal field. Defaults to 45.
    """
    single_dim = np.random.random()>0.5
    thresh = np.random.uniform(0.05,1,2)
    lam = np.random.uniform(1,10,2)
    n = 2*width
    sinusoid = _sample_1d_sinusoid(n)

    # central width x width crop of the final rotation: rows/cols [c0, c1)
    c0 = width//2
    c1 = c0 + width
    M2 = rotation_inverse((n, n), angle2)
    sr0, sr1, sc0, sc1 = source_window(M2, (c0, c1), c1, (n, n))

    # intermediate field on rows [sr0, sr1), cols [0, sc1)
    M = rotation_inverse((n, n), angle)
    r0, r1, q0, q1 = source_window(M, (sr0, sr1), sc1, (n, n))
    src = np.broadcast_to(sinusoid[q0:q1], (r1 - r0, q1 - q0))
    field = warp_window(src, (r0, q0), M, (sr0, sr1), sc1)
    sinusoid_transfer(field, thresh[0], lam[0], field)

    if not(single_dim):
        sinusoid_y = _sample_1d_sinusoid(n)
        M1 = rotation_inverse((n, n), angle1)
        r0, r1, q0, q1 = source_window(M1, (sr0, sr1), sc1, (n, n))
        src = np.broadcast_to(sinusoid_y[r0:r1, None], (r1 - r0, q1 - q0))
        field_y = warp_window(src, (r0, q0), M1, (sr0, sr1), sc1)
        sinusoid_transfer(field_y, thresh[1], lam[1], field_y)
        field *= field_y

    sinusoid = warp_window(field[:, sc0:sc1], (sr0, sc0), M2, (c0, c1), c1)[:, c0:]
    sinusoid = normalize(sinusoid)

    return(sinusoid)


def sample_interpolation_map(mixing_types = ["sin"],width = 1000,warp = True):
    """Product of the interpolation maps listed in mixing_types ("sin", "grid"),
    each optionally warped. Returns a width x width map with values in [0, 1]."""
    maps = []
    if "sin" in mixing_types:
        angle = np.random.uniform(-45,45)
        angle1 = angle+np.random.choice([-1,1])*np.random.uniform(15,45)
        angle2 = np.random.uniform(-22.5,22.5)
        #ad hoc proportion

        sin = sample_sinusoid(width = width,angle = angle,angle1 = angle1,angle2 = angle2)

        if warp:
            sin = np.clip(generate_perturbation(sin),0,1)
        maps.append(sin)

    if "grid" in mixing_types:
        #ad hoc not justified
        angle_grid = np.random.uniform(-45,45)
        #ad hoc proportion
        two_dim = np.random.random()>0.3
        if two_dim:
            #ad hoc not justified
            period_grid = [np.random.randint(20,100),np.random.randint(20,100)]
        else:
            #ad hoc not justified
            period_grid = [np.random.randint(20,100)]
        #ad hoc not justified
        grid = sample_grid(width = width,period = period_grid,angle = angle_grid)
        if warp:
            grid = np.clip(generate_perturbation(grid),0,1)
        # the original computes grid * sin
        maps.insert(0, grid)

    if not maps:
        return np.ones((width,width))
    res = maps[0].astype(np.float64, copy=len(maps) == 1)
    for m in maps[1:]:
        res = res*m
    return(res)
