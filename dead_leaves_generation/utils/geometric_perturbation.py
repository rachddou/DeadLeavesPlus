import numpy as np
import cv2
from numba import njit
 
@njit
def generate_triangular_pattern(shape,T):
    tmp = shape/T
    n_period = int(tmp)
    start = 0
    pattern = np.zeros(shape, dtype = np.float32)
    for i in range(n_period):
        for j in range(T//2):
            if i*T+j == shape:
                return(pattern)
            else:
                pattern[i*T+j] = 4*j/T -1.
        for j in range(T//2):
            if i*T+T//2 + j == shape:
                return(pattern)
            else:
                pattern[i*T+T//2 + j] = 4*(T//2-j)/T -1.
    return(pattern)

def generate_vector_field(shape):
    """
    function that generates the direction maps for the distortion parameter
    input :
    - shape : is the side length of the square

    output :
    - u the direction map wrt x
    - v the direction map wrt y
        """
    u,v = np.zeros((shape,shape)),np.ones((shape,shape))
    # type = np.random.choice(["athmos","sin","triangle"],p=[0.33,0.34,0.33])
    type = "athmos"
    if type == "athmos":
        u = np.random.normal(0.,1.,(shape,shape))
        v = np.random.normal(0.,1.,(shape,shape))
        
        s = np.random.randint(10,20)
        t = np.random.randint(5,s//2+1)
        u = cv2.GaussianBlur(u,(61,61),sigmaX = s, borderType = cv2.BORDER_DEFAULT)
        v = cv2.GaussianBlur(v,(61,61),sigmaX = s, borderType = cv2.BORDER_DEFAULT)
        u = (u-np.mean(u))*(t/np.std(u))
        v = (v-np.mean(v))*(t/np.std(v))
    elif type == "sin":
        #ad hoc not justified
        T = np.random.randint(50,1000)
        intensity = np.random.uniform(0.8,0.1*T*1.2)
        sinusoid = intensity*np.sin(2*np.pi*np.arange(0,shape)/T).reshape((shape,1))
        u = np.repeat(sinusoid,shape,axis = -1)
    elif type == "triangle":
        #ad hoc  not justified
        T = np.random.randint(50,1000)
        intensity =np.random.uniform(0.1*T,0.5*T)
        pattern = generate_triangular_pattern(shape,T).reshape((shape,1))
        pattern = intensity*pattern
        u = np.repeat(pattern,shape,axis = -1)
    return(u,v)

def bilinear_interpolate(im, xx, yy):
    """
    bilinear interpolation function
    input:
    -im : the image to interpolate
    -x : the interpolation parameters wrt to x direction
    -y : the interpolation parameters wrt to y direction

    output :
    - the bilinear interpolation
    """
    x0 = np.floor(xx).astype(int)
    x1 = x0 + 1
    y0 = np.floor(yy).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1-xx) * (y1-yy)
    wb = (x1-xx) * (yy-y0)
    wc = (xx-x0) * (y1-yy)
    wd = (xx-x0) * (yy-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def generate_perturbation(x):
    """
    function that generates the actual distortion perturbation
    input :
    - x : the original image Grey level image
    - s : smoothness parameter
    - t : transport parameter
    - type : type of perturbation to apply
    output :
    the distorted image

    USAGE : change parameters s and t in this function
    """
    shape = x.shape[0]
    u,v = generate_vector_field(shape)
    xx, yy = np.meshgrid(np.arange(shape), np.arange(shape))  # cartesian indexing
    res = np.zeros(x.shape)

    res[:,:] = bilinear_interpolate(x[:,:], u+xx, v+yy)+np.min(x)
    return(res)