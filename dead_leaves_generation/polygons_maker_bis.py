import numpy as np
import rasterio.features
from skimage.filters import gaussian
import shapely
import math
import random
import cv2



def rotate_image(img, angle):
    size_reverse = np.array(img.shape[1::-1]) # swap x with y
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.)
    MM = np.absolute(M[:,:2])
    size_new = MM @ size_reverse
    M[:,-1] += (size_new - size_reverse) / 2.
    return cv2.warpAffine(img, M, tuple(size_new.astype(int)))

def make_rectangle_mask(radius):
    """Generates a binary mask of a rectangle of area = pi*radius^2.

    Args:
        radius (_type_): radius sampled from a power law distribution.
    """
    area = np.pi*(radius**2)
    ratio = random.uniform(0.5,1)
    width = int(math.sqrt(ratio*area))
    length = int(math.sqrt(area/ratio))
    mask = np.zeros((length,length))
    mask[int(length//2-width//2):int(length//2+width//2),:] +=1

    angle = random.uniform(0,180)
    mask = np.bool_(rotate_image(mask,angle))
    mask = np.bool_(mask)
    h,w = mask.shape
    h_odd = 2*(h//2)-1
    w_odd = 2*(w//2)-1

    mask = mask[:h_odd,:w_odd]
    return(mask)

def sample_points_circle(n,radius):
    """Generates n points sampled uniformly in a circle of radius radius.

    Args:
        n (int): number of points to generate.
        radius (int): radius of the circle.
    """
    # distance_to_center = np.sqrt(np.random.uniform(0, radius**2, n))
    distance_to_center = np.random.randint(0,radius,n)
    angle = np.random.uniform(-np.pi,np.pi,n)
    
    x = np.array([radius+distance_to_center[i]*np.cos(angle[i]) for i in range(n)])
    y = np.array([radius+distance_to_center[i]*np.sin(angle[i]) for i in range(n)])
    return(np.stack([x,y],axis = -1))


def binary_polygon_generator(size, n = 100, concavity = 0.3, allow_holes = True, smoothing = True):
    """Generates a binary image of a polygon with a concave hull.

    Args:
        size (_type_): size of the final binary image.
        n (int, optional): number of points to generate the concave hull. Defaults to 100.
        concavity (float, optional): coefficient alpha of the alpha shape. Defaults to 0.3.
        allow_holes (bool, optional): allow for holes in the shape or not. Defaults to True.
        smoothing (bool, optional): whether or not applying gaussian smoothing. Defaults to True.
    """

    coords = sample_points_circle(n,size//2)

    points = shapely.MultiPoint([(coords[k,0],coords[k,1]) for k in range(n)])

    concave_hull = shapely.concave_hull(points, ratio=concavity, allow_holes=allow_holes)

    img = rasterio.features.rasterize([concave_hull], out_shape=(size, size)).astype(np.bool_)
    
    if smoothing:
        min_blur = max(0.5,2*(size/200))
        max_blur = min(10,5*(size/200))
        blur_val = np.random.uniform(min_blur,max_blur)
        img = gaussian(img.astype(np.float32),blur_val)
        res = np.zeros(img.shape,dtype=np.bool_)
        res[img>0.5] = True
        return(res)
    else:
        return(img)
    
    
    
if __name__ == "__main__":
    import skimage.io as skio
    size = 400
    for i in range(200):
        n = random.randint(30,150)
        concavity = np.random.uniform(0.2,0.5)
        allow_holes = random.choice([True,False])
        smoothing = random.choice([True,False])
        binary_image = 1-binary_polygon_generator(size, n = n, concavity = concavity, allow_holes = allow_holes, smoothing = smoothing)

        binary_image = np.uint8(binary_image*255)
        skio.imsave(f'../polygons/polygon_{i}.png',binary_image)