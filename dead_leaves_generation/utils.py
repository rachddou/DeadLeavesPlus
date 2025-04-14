import numpy as np
from skimage.color import rgb2lab, lab2rgb
from numba import njit
import cv2
from PIL import Image
from time import time
import skimage.io as skio



def rotate_CV(image, angle ):
    """Rotate an image by a given angle using OpenCV

    Args:
        image (numpy array): input image
        angle (float): angle of rotation in degrees

    Returns:
        numpy array: rotated image
    """
    #in OpenCV we need to form the tranformation matrix and apply affine calculations
    #
    h,w = image.shape[:2]
    cX,cY = (w//2,h//2)
    M = cv2.getRotationMatrix2D((cX,cY),angle,1)
    rotated = cv2.warpAffine(image,M , (w,h),flags=cv2.INTER_LINEAR)
    return rotated



def normalize(x):
    """
    function that normalizes an image between 0 and 1

    Args:
        x numpy array: input image
    """
    x = (x-x.min())/(x.max()-x.min())
    return(x)


def theoretical_number_disks(delta,r_min,r_max,width,alpha):
    """function that computes the theoretical number of disks in a dead leaves image with given parameters.

    Args:
        delta (float): percentage of the image covered by the disks
        r_min (int): minimal radius of the disks
        r_max (int): maximal radius of the disks
        width (int): width of the image
        alpha (float): power law parameter 
    """
    d = 0
    e =  alpha - 1
    for k in range(r_min,r_max,1):
        d+=(1/((1/r_min**e) -(1/r_max**e))*(1-((k/(k+1))**2)))
    d*=(np.pi)/(width**2)
    return(np.log(delta)/np.log(1-d))

def N_sup_r(r,width,r_min,r_max):
    delta = 100/(width**2)
    N = theoretical_number_disks(delta,r_min,r_max,width)
    prop = ((1/r**2) - (1/r_max**2))/(1/(r_min**2) - 1/(r_max**2))
    res = N*prop
    return(res)


def make_grid(width = 100,period = [100],thickness = 3 ,angle = 45):
    """function that creates a grid pattern with a given orientation

    Args:
        width (int, optional): width of the final image. Defaults to 100.
        period (list, optional): period of the grid. If len(period) is 2, then the grid is bi-directional. Defaults to [100].
        thickness (int, optional): thickness of the grid's borders. Defaults to 3.
        angle (int, optional): orientation of the grid. Defaults to 45.
    """
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

def make_sinusoid(width = 100,angle = 45 ,angle1 = 45,angle2 = 45,variable_freq = False):
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

def pattern_patch_two_colors(color_1,color_2,width=100,period=[100],thickness = 3, angle=45,warp = False,type = "sin"):
    """function that mixes two color maps with a sinusoidal pattern.

    Args:
        color_1 (_type_): either an image or a single RGB color
        color_2 (_type_): either an image or a single RGB color
        width (int, optional): image width. Defaults to 100.
        period (list, optional): period of the oscillations. Defaults to [100].
        thickness (int, optional): thickness of the grid. Defaults to 3.
        angle (int, optional): rotation angle. Defaults to 45.
        warp (bool, optional): Whether or not apply athmospheric perturbation. Defaults to False.
        type (str, optional): define the type of interpolation map. Defaults to "sin".
    """
    # functions that create a sinusoidal patch between 2 colors with a random orientation.
    # parameters:
    # colors : (int)
    # width : (int) size of the patch
    # period : (list) list of periods of the sinusoid, if len  = 2 then bi-directional sinus otherwise single direction
    # angle : (int) angle in degree
    
    res_image = np.zeros((width,width))

    color_1 = color_1/255.
    color_2 = color_2/255.
    color_transform_1 = rgb2lab(color_1)
    color_transform_2 = rgb2lab(color_2)
    
    pattern = np.zeros((width,width))

    if type == "sin":
        pattern = make_sinusoid(width = width,angle = angle,variable_freq=(np.random.random()>0.5))
    elif type == "grid":
        pattern = make_grid(width = width,period = period,angle = angle ,thickness = thickness)


    if warp:        
        pattern = np.clip(generate_perturbation(pattern),0,1)  


    r = pattern*(color_transform_2[...,0]-color_transform_1[...,0])+color_transform_1[...,0]
    g = pattern*(color_transform_2[...,1]-color_transform_1[...,1])+color_transform_1[...,1]
    b = pattern*(color_transform_2[...,2]-color_transform_1[...,2])+color_transform_1[...,2]

    res_image = np.stack([r,g,b],axis = -1)

    res_image = lab2rgb(res_image)

    res_image = np.uint8(res_image*255)

    return(res_image)

def logistic(x, L=1, x_0=0, k=1):
    """
    logistic function

    Args:
        x (_type_): input
        L (int, optional): _description_. Defaults to 1.
        x_0 (int, optional): _description_. Defaults to 0.
        k (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: y = L / (1 + exp(-k * (x - x_0)))
    """
    return L / (1 + np.exp(-k * (x - x_0)))

def normalized_logistic(x,k):
    """
    function that normalizes a logistic function between 0 and 1
    """
    y = (logistic(12.1*x-6,k=k)-logistic(-6,k=k))/(logistic(6.1,k=k)-logistic(-6,k=k))
    return(y)

def linear_color_gradient(color_1,color_2,width,angle, k = 0.5,color_space = "lab"):
    """function that creates a color gradient between two colors in different color space.
    Args:
        color_1 (_type_): first color in RGB
        color_2 (_type_): second color in RGB
        width (_type_): width of the gradient
        angle (_type_): angle of the gradient
        k (float, optional): smoothness factor of the gradient. Defaults to 0.5.
        color_space (str, optional): _description_. Defaults to "lab".
    """
    lin_gradient =np.tile(np.linspace(0,1,2*width),(2*width,1))
    rotated_grad = normalize(rotate_CV(lin_gradient,angle)[width//2:-width//2,width//2:-width//2])

    lin_gradient = normalized_logistic(rotated_grad,k=k)

    color_1 = color_1/255.
    color_2 = color_2/255.

    if color_space == "rgb":
        color_transform_1 =  color_1
        color_transform_2 =  color_2
    if color_space == "lab":
        color_transform_1 =  rgb2lab(color_1)
        color_transform_2 =  rgb2lab(color_2)

    x = color_transform_1[0]+(color_transform_2[0]-color_transform_1[0])*lin_gradient
    y = color_transform_1[1]+(color_transform_2[1]-color_transform_1[1])*lin_gradient
    z = color_transform_1[2]+(color_transform_2[2]-color_transform_1[2])*lin_gradient
    final_img = np.stack([x,y,z],axis = -1)

    if color_space == "lab":
        final_img = np.uint8(255*lab2rgb(final_img))
    return(final_img)

def sigmoid(x,lamda):
    """
    sigmoid function to transform the sinusoid in a sharper transition
    """
    sig = 1/(1+np.exp(-lamda*(x)))
    if lamda >=20:
        sig[sig>0.5] = 1
        sig[sig<=0.5] = 0
    return(sig)


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

def random_phase_im(image):
    """apply a random phase to the image in the frequency domain

    Args:
        image (_type_): input RGB image
    """
    image = image /255
    fft_3 = [np.fft.fft2(image[...,i]) for i in range(3)]
    final_im = np.zeros((fft_3[0].shape[0],fft_3[0].shape[1],3))
    spec = [abs(fft_3[i]) for i in range(3)]
    phase = [np.angle(fft_3[i]) for i in range(3)]
    random_phase = np.random.uniform(0,2*np.pi,(image.shape[0],image.shape[1]))
    new_phase = [phase[i]+random_phase for i in range(3)]
    #new_phase = [phase[i]%(2*np.pi) - np.pi for i in range(3)]
    for i in range(3):
        final_fft_c = spec[i]*np.exp(1j*new_phase[i])
        final_c =np.fft.ifft2(final_fft_c)
        final_c = np.abs(final_c)
        final_im[...,i] = final_c
        
    final_im = np.clip(final_im,0,1)
    final_im = np.uint8(255*final_im)
    return(final_im)

def sample_magnitude(resolution,slope):
    """function that creates a magnitude map for the frequency domain in 1/f^slope

    Args:
        resolution (int): size of the texture image
        slope (float): slope of the magnitude map in the log domain
    """
    # slope_0, slope_1 = 0.5,3.5
    # d = np.random.normal() * np.abs(slope_1 - slope_0) / 15 / 4

    # slope_x = slope + d
    # slope_y = slope - d

    fx, fy = np.meshgrid(range(resolution), range(resolution))

    fx = fx - resolution / 2
    fy = fy - resolution / 2

    # fr = 1e-16 + np.abs(fx / resolution) ** slope_x + + np.abs(fy / resolution) ** slope_y
    
    fr = 1e-16 + np.abs(fx / resolution) ** slope +  np.abs(fy / resolution) ** slope

    magnitude = 1 / np.fft.fftshift(fr)
    magnitude[0,0] = 0
    magnitude = np.tile(magnitude[:,:,None], (1,1,3))
    return(magnitude)

def freq_noise(image,width,slope):
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
    index =np.random.choice(np.arange(0,p*p),replace= True,size = width*width)
    

    image = image[index]
    image = image.reshape((width,width,3)).astype(np.float32)
    magnitude = sample_magnitude(width,slope)

    for c in range(3):
        x = image[...,c]

        m = np.mean(x)
        s = np.std(x)

        fft_img = np.fft.fft2(x - m)
        fft_phase = np.angle(fft_img)

        fft_imposed = magnitude[...,c] * np.exp(1j * fft_phase)
        y = np.real(np.fft.ifft2(fft_imposed))

        y = y - y.mean()
        y = y / np.std(y) * s + m

        image[...,c] = y
    image = np.uint8(np.clip(image,0,255))
    return(image)

def thresholding(img,a,b):
    """
    soft threshold of value min =a and max =b
    """
    img[img<a] = a
    img[img>b] = b
    img = np.uint8(255*(img-a)/(b-a))
    return(img)

def shadow_gradient(texture,angle, k = 0.5,low =0.2):
    """
    function that creates a color gradient image between two colors in different color space.
    This is done to study the impact of the color space to the aspect of the gradient.
    """
    width = texture.shape[0]
    lin_gradient =np.tile(np.linspace(0,1,2*width),(2*width,1))
    rotated_grad = normalize(rotate_CV(lin_gradient,angle)[width//2:-width//2,width//2:-width//2])

    lin_gradient = low  +  (1-low)*normalized_logistic(rotated_grad,k=k)
    x = texture[...,0]*lin_gradient
    y = texture[...,1]*lin_gradient
    z = texture[...,2]*lin_gradient
    final_img = np.uint8(np.stack([x,y,z],axis = -1))
    return(final_img)


def mixing_materials_v2(tmp1 = np.random.randint(0,255,(100,100,3)),tmp2 = np.random.randint(0,255,(100,100,3)),single_color1 = True,single_color2 = True,mixing_types = ["sin"],width = 1000,thresh_val = 10,warp = True):
    """function that mixes two color/texture maps with either a sinusoidal pattern, a grid pattern or a noise pattern.

    Args:
        tmp1 (_type_, optional): source color image 1. Defaults to np.random.randint(0,255,(100,100,3)).
        tmp2 (_type_, optional): source color image 2. Defaults to np.random.randint(0,255,(100,100,3)).
        single_color1 (bool, optional): whether texture 1 is a single color or a colored noise map. Defaults to True.
        single_color2 (bool, optional): whether texture 2 is a single color or a colored noise map. Defaults to True.
        mixing_types (list, optional): interpolation methods between two textures. Defaults to ["sin"].
        width (int, optional): size of the final texture image. Defaults to 1000.
        thresh_val (int, optional): specific to noise mixing, the thresholding parameter for binary masking. Defaults to 10.
        warp (bool, optional): whether or not to apply warping for the sinusoidal interpolation maps. Defaults to True.
    """
    if single_color1:
        color1 = tmp1[np.random.randint(0,tmp1.shape[0],1),np.random.randint(0,tmp1.shape[1],1),:].reshape((1,1,3))

    else:
        slope1 = np.random.uniform(0.5,2.5)
        color1 = freq_noise(tmp1,width,slope1)
    if single_color2:
        color2 = tmp2[np.random.randint(0,tmp2.shape[0],1),np.random.randint(0,tmp2.shape[1],1),:].reshape((1,1,3))

    else:
        #ad hoc ok
        slope2 = np.random.uniform(0.5,2.5)
        color2 =freq_noise(tmp2,width,slope2)

    if "sin" in mixing_types:
        angle = np.random.uniform(-45,45)
        angle1 = angle+np.random.choice([-1,1])*np.random.uniform(15,45)
        angle2 = np.random.uniform(-22.5,22.5)
        #ad hoc proportion
        
        sin = make_sinusoid(width = width,angle = angle,angle1 = angle1,angle2 = angle2, variable_freq=(np.random.random()>0.5))
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
        thickness = np.random.randint(1,2)
        grid = make_grid(width = width,period = period_grid,angle = angle_grid ,thickness = thickness)
        if warp:
            grid = np.clip(generate_perturbation(grid),0,1)
    else:
        grid = np.ones((width,width))


    if "noise" in mixing_types:
        
        pattern = np.random.randint(0,255,(width,width,3))
        #ad hoc ok
        slope_mixing = np.random.uniform(1.5,3)
        pattern = np.mean(np.uint8(np.clip(freq_noise(pattern,width,slope_mixing),0,255)),axis =2)
        pattern = thresholding(pattern,128-thresh_val,128+thresh_val)/255.
    else:
        pattern = np.ones((width,width))
    pattern = grid*sin*pattern    
    color1 = color1/255.
    color2 = color2/255.
    color_transform_1 = rgb2lab(color1)
    color_transform_2 = rgb2lab(color2)
    r = pattern*(color_transform_2[...,0]-color_transform_1[...,0])+color_transform_1[...,0]
    g = pattern*(color_transform_2[...,1]-color_transform_1[...,1])+color_transform_1[...,1]
    b = pattern*(color_transform_2[...,2]-color_transform_1[...,2])+color_transform_1[...,2]

    res_image = np.stack([r,g,b],axis = -1)

    res_image = lab2rgb(res_image)

    res_image = np.uint8(res_image*255)
    return(res_image)


def find_coeffs(pa, pb):
    """function that finds the coefficients for the perspective function.

    Args:
        pa (_type_): _description_
        pb (_type_): _description_

    Returns:
        _type_: coefficients for the perspective function
    """
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float32)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def perspective_shift(image):
    """function that applies a perspective shift to an image

    Args:
        image (_type_): input image
    """

    size = image.shape[0]
    img = Image.fromarray(image)
    alpha =  2
    beta  = int(size//4)
    mode = np.random.randint(0,3)

    coeffs = find_coeffs(
        [(0, 0), (0, size), (size//alpha, size -  beta), (size//alpha, beta)],
            [(0, 0), (0,size), (size,size), (size,0)]
            )
    img = img.transform((size, size), Image.PERSPECTIVE, coeffs,
            Image.BICUBIC)

    img = np.array(img)
    result = img[beta:size-beta,0:size//alpha]
    if mode == 0 :
        result = np.fliplr(result)
    elif mode == 1 :
        result = np.rot90(result)
    return(result)