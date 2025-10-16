import numpy as np
from dead_leaves_generation.utils.utils import rotate_CV, normalize, sigmoid,thresholding
from dead_leaves_generation.utils.geometric_perturbation import generate_perturbation
from dead_leaves_generation.utils.colored_noise import sample_color_noise
from numba import njit

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
    grid = np.tile(x,(2*width,1))

    if len(period)==2:
        y =  np.ones((2*width))
        for i in range(1,1+(2*width)//(period[1]+thickness)):
            y[i*(period[1]+thickness)-thickness:i*(period[1]+thickness)] = 0
        grid_y = np.tile(y,(2*width,1)).T
        grid  = grid*grid_y
    grid = normalize(grid[width//2:-width//2,width//2:-width//2])
    grid = rotate_CV(grid,angle)
    return(grid)

@njit
def sample_period(T_min,T_max,n_period):
    periods = np.floor(T_min + (T_max-T_min)*np.random.power(1/2.5, size=n_period))
    return(periods)
@njit
def variable_oscillations(width,T_min,T_max,n_freq):
    """function that creates a pseudo-periodic pattern with variable frequencies in 1D.

    Args:
        width (_type_): width of the final image
        T_min (_type_): minimal period of the pattern
        T_max (_type_): maximal period of the pattern
        n_freq (_type_): length of the frequency array
    """
    
    freq_cycles = sample_period(T_min,T_max,n_freq)
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

def sample_sinusoid(width = 100,angle = 45 ,angle1 = 45,angle2 = 45,variable_freq = False):
    """function that create a pseudo-periodic grey-level pattern based on sinusoidal functions.
    This patterns then serves as an interpolation map either between two colors or two texture maps.

    Args:   
        width (int, optional): width of the final image. Defaults to 100.
        period (list, optional): . Defaults to [100].
        angle (int, optional): angle for dimension 1. Defaults to 45.
        angle1 (int, optional): angle for dimension 2. Defaults to 45.
        angle2 (int, optional): rotation applied to the whole sinusoidal field. Defaults to 45.
        variable_freq (bool, optional): Creates a sequence of single periods of random length. Defaults to False.
    """
    T_min = 5
    T_max = 50
    single_dim = np.random.random()>0.5
    if variable_freq:
        sinusoid =  variable_oscillations(2*width,T_min,T_max,20)
    else:
        period = sample_period(T_min,T_max,1)
        sinusoid = np.sin(((2*np.pi)/period[0])*np.arange(0,2*width))
    

    sinusoid = rotate_CV(np.tile(sinusoid,(2*width,1)),angle)

    if not(single_dim):
        if variable_freq:
            sinusoid_y =  variable_oscillations(2*width,T_min,T_max,20)
        else:
            period = sample_period(T_min,T_max,1)
            sinusoid_y = np.sin(((2*np.pi)/period[0])*np.arange(0,2*width))
        sinusoid_y = np.tile(sinusoid_y,(2*width,1)).T
        sinusoid_y = rotate_CV(sinusoid_y,angle1)
        sinusoid  = sinusoid*sinusoid_y


    #ad hoc ok
    lamda = np.random.uniform(1,10)
    sinusoid = sigmoid(sinusoid,lamda)
    sinusoid = (sinusoid-sigmoid(np.array([0]),lamda))/(sigmoid(np.array([1]),lamda)-sigmoid(np.array([0]),lamda))

    sinusoid = 0.5+ 0.5*sinusoid
    sinusoid = normalize(rotate_CV(sinusoid,angle2)[width//2:-width//2,width//2:-width//2])

    return(sinusoid)


def sample_interpolation_map(mixing_types = ["sin"],width = 1000,thresh_val = 10,warp = True):
    if "sin" in mixing_types:
        angle = np.random.uniform(-45,45)
        angle1 = angle+np.random.choice([-1,1])*np.random.uniform(15,45)
        angle2 = np.random.uniform(-22.5,22.5)
        #ad hoc proportion
        
        sin = sample_sinusoid(width = width,angle = angle,angle1 = angle1,angle2 = angle2, variable_freq=(np.random.random()>0.5))
        if warp:
            sin = np.clip(generate_perturbation(sin),0,1)
    else:
        sin = np.ones((width,width))
    
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
        grid = np.ones((width,width))


    if "noise" in mixing_types:
        
        pattern = np.random.randint(0,255,(width,width,3))
        #ad hoc ok
        slope_mixing = np.random.uniform(1.5,3)
        pattern = np.mean(np.uint8(np.clip(sample_color_noise(pattern,width,slope_mixing),0,255)),axis =2)
        pattern = thresholding(pattern,128-thresh_val,128+thresh_val)/255.
    else:
        pattern = np.ones((width,width))
    pattern = grid*sin*pattern
    return(pattern)