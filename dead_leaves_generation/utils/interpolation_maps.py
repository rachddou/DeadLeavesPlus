import numpy as np
from dead_leaves_generation.utils.utils import (rotate_CV, normalize, sigmoid, thresholding,
                                                rotation_inverse, source_window, warp_window)
from dead_leaves_generation.utils.geometric_perturbation import generate_perturbation
from dead_leaves_generation.utils.colored_noise import sample_color_noise
from dead_leaves_generation.utils.kernels import sinusoid_transfer
from numba import njit
TMIN = 5
TMAX = 100
def sample_grid(width = 100,period = [100],angle = 45):
    """function that creates a grid pattern with a given orientation

    Args:
        width (int, optional): width of the final image. Defaults to 100.
        period (list, optional): period of the grid. If len(period) is 2, then the grid is bi-directional. Defaults to [100].
        thickness (int, optional): thickness of the grid's borders. Defaults to 3.
        angle (int, optional): orientation of the grid. Defaults to 45.
    """
    thickness =  np.random.randint(1,3)
    x = np.ones((2*width))
    for i in range(1,1+(2*width)//(period[0]+thickness)):
        x[i*(period[0]+thickness)-thickness:i*(period[0]+thickness)] = 0
    crop = slice(width//2, -width//2)
    xc = x[crop]
    grid = np.broadcast_to(xc, (len(xc), len(xc)))

    if len(period)==2:
        y =  np.ones((2*width))
        for i in range(1,1+(2*width)//(period[1]+thickness)):
            y[i*(period[1]+thickness)-thickness:i*(period[1]+thickness)] = 0
        grid  = grid*y[crop, None]
    grid = normalize(grid)
    grid = rotate_CV(grid,angle)
    return(grid)

@njit(cache=True)
def sample_period(n_period):
    # periods = np.floor(TMIN + (TMAX-TMIN)*np.random.power(1/2.5, size=n_period))
    periods = np.floor(TMIN + (TMAX-TMIN)*np.random.random(size=n_period))
    return(periods)
@njit(cache=True)
def variable_oscillations(width,n_freq):
    """function that creates a pseudo-periodic pattern with variable frequencies in 1D.

    Args:
        width (_type_): width of the final image
        n_freq (_type_): length of the frequency array
    """

    freq_cycles = sample_period(n_freq)
    freq_cycles_bis = np.array([int(freq_cycles[i]) for i in range(n_freq)])
    T_full_cycle = np.sum(freq_cycles_bis)
    N_cycles = width//T_full_cycle
    res = np.zeros(width,dtype = np.float32)
    start = 0
    for n in range(N_cycles+1):
        for i in range(n_freq):
            period = freq_cycles_bis[i]
            for j in range (period):
                if start+j == width:
                    return(res)
                else:
                    res[start+j] = np.sin(((2*np.pi)/period)*j)
            start+=period
    return(res)

def _sample_1d_sinusoid(n):
    period = sample_period(1)
    return np.sin(((2*np.pi)/period[0])*np.arange(0,n))

def _sample_variable_sinusoid(width, single_dim, thresh, lamda, angle, angle1, angle2):
    """Full-canvas version of sample_sinusoid, kept for variable frequencies:
    their float32 maps go through OpenCV's float32 warp, whose result depends on
    the canvas position, so the windowed computation would not be identical."""
    sinusoid = rotate_CV(np.tile(variable_oscillations(2*width,20),(2*width,1)),angle)
    sinusoid = 0.5+ 0.5*sinusoid
    sinusoid[sinusoid>thresh[0]] = 1
    sinusoid = sigmoid(sinusoid,lamda[0])
    sinusoid = (sinusoid-sigmoid(np.array([0]),lamda[0]))/(sigmoid(np.array([1]),lamda[0])-sigmoid(np.array([0]),lamda[0]))
    if not(single_dim):
        sinusoid_y = rotate_CV(np.tile(variable_oscillations(2*width,20),(2*width,1)).T,angle1)
        sinusoid_y = 0.5+ 0.5*sinusoid_y
        sinusoid_y[sinusoid_y>thresh[1]] = 1
        sinusoid_y = sigmoid(sinusoid_y,lamda[1])
        sinusoid_y = (sinusoid_y-sigmoid(np.array([0]),lamda[1]))/(sigmoid(np.array([1]),lamda[1])-sigmoid(np.array([0]),lamda[1]))
        sinusoid  = sinusoid*sinusoid_y
    return normalize(rotate_CV(sinusoid,angle2)[width//2:-width//2,width//2:-width//2])

def sample_sinusoid(width = 100,angle = 45 ,angle1 = 45,angle2 = 45,variable_freq = False):
    """function that create a pseudo-periodic grey-level pattern based on sinusoidal functions.
    This patterns then serves as an interpolation map either between two colors or two texture maps.

    The pattern is built on a (2*width)^2 canvas, rotated, and the central
    width^2 crop is kept. Only the windows of the intermediate images that
    contribute to that crop are computed (see utils.warp_window).

    Args:
        width (int, optional): width of the final image. Defaults to 100.
        period (list, optional): . Defaults to [100].
        angle (int, optional): angle for dimension 1. Defaults to 45.
        angle1 (int, optional): angle for dimension 2. Defaults to 45.
        angle2 (int, optional): rotation applied to the whole sinusoidal field. Defaults to 45.
        variable_freq (bool, optional): Creates a sequence of single periods of random length. Defaults to False.
    """
    single_dim = np.random.random()>0.5
    thresh = np.random.uniform(0.05,1,2)
    lamda = np.random.uniform(1,10,2)
    if variable_freq:
        return _sample_variable_sinusoid(width, single_dim, thresh, lamda, angle, angle1, angle2)
    n = 2*width
    sinusoid = _sample_1d_sinusoid(n)

    # central crop of the final rotation: rows/cols [c0, c1)
    c0, c1 = width//2, n + (-width)//2
    M2 = rotation_inverse((n, n), angle2)
    sr0, sr1, sc0, sc1 = source_window(M2, (c0, c1), c1, (n, n))

    # intermediate field on rows [sr0, sr1), cols [0, sc1)
    M = rotation_inverse((n, n), angle)
    r0, r1, q0, q1 = source_window(M, (sr0, sr1), sc1, (n, n))
    src = np.broadcast_to(sinusoid[q0:q1], (r1 - r0, q1 - q0))
    field = warp_window(src, (r0, q0), M, (sr0, sr1), sc1)
    sinusoid_transfer(field, thresh[0], lamda[0], field)

    if not(single_dim):
        sinusoid_y = _sample_1d_sinusoid(n)
        M1 = rotation_inverse((n, n), angle1)
        r0, r1, q0, q1 = source_window(M1, (sr0, sr1), sc1, (n, n))
        src = np.broadcast_to(sinusoid_y[r0:r1, None], (r1 - r0, q1 - q0))
        field_y = warp_window(src, (r0, q0), M1, (sr0, sr1), sc1)
        sinusoid_transfer(field_y, thresh[1], lamda[1], field_y)
        field *= field_y

    sinusoid = warp_window(field[:, sc0:sc1], (sr0, sc0), M2, (c0, c1), c1)[:, c0:]
    sinusoid = normalize(sinusoid)

    return(sinusoid)


def sample_interpolation_map(mixing_types = ["sin"],width = 1000,thresh_val = 10,warp = True):
    if "sin" in mixing_types:
        angle = np.random.uniform(-45,45)
        angle1 = angle+np.random.choice([-1,1])*np.random.uniform(15,45)
        angle2 = np.random.uniform(-22.5,22.5)
        #ad hoc proportion

        # sin = sample_sinusoid(width = width,angle = angle,angle1 = angle1,angle2 = angle2, variable_freq=(np.random.random()>0.5))
        sin = sample_sinusoid(width = width,angle = angle,angle1 = angle1,angle2 = angle2, variable_freq=False)

        if warp:
            sin = np.clip(generate_perturbation(sin),0,1)
    else:
        sin = None

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
    else:
        grid = None


    if "noise" in mixing_types:

        pattern = np.random.randint(0,255,(width,width,3))
        #ad hoc ok
        slope_mixing = np.random.uniform(1.5,3)
        pattern = np.mean(np.uint8(np.clip(sample_color_noise(pattern,width,slope_mixing),0,255)),axis =2)
        pattern = thresholding(pattern,128-thresh_val,128+thresh_val)/255.
    else:
        pattern = None

    # product of the selected maps (missing maps are all-ones in the reference)
    maps = [m for m in (grid, sin, pattern) if m is not None]
    if not maps:
        return np.ones((width,width))
    res = maps[0].astype(np.float64, copy=len(maps) == 1)
    for m in maps[1:]:
        res = res*m
    return(res)
