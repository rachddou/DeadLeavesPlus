import cv2
import numpy as np
from skimage.color import rgb2lab, lab2rgb


## rotation function using OpenCV
def rotate_CV(image, angle ):
    """
    Rotate an image by a given angle using OpenCV

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


# normalizing/ thresholding function
def normalize(x):
    """
    function that normalizes an image between 0 and 1

    Args:
        x numpy array: input image
    """
    x = (x-x.min())/(x.max()-x.min())
    return(x)

def thresholding(img,a,b):
    """
    soft threshold of value min =a and max =b
    """
    img[img<a] = a
    img[img>b] = b
    img = np.uint8(255*(img-a)/(b-a))
    return(img)

## functions for counting remaining disks
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
    if d>=1.0:
        return(20)
    else:
        return(np.log(delta)/np.log(1-d))

def N_sup_r(r,width,r_min,r_max):
    delta = 100/(width**2)
    N = theoretical_number_disks(delta,r_min,r_max,width)
    prop = ((1/r**2) - (1/r_max**2))/(1/(r_min**2) - 1/(r_max**2))
    res = N*prop
    return(res)


## logit functions
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

def sigmoid(x,lamda):
    """
    sigmoid function to transform the sinusoid in a sharper transition
    """
    sig = 1/(1+np.exp(-lamda*(x)))
    if lamda >=20:
        sig[sig>0.5] = 1
        sig[sig<=0.5] = 0
    return(sig)

## color gradient function

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