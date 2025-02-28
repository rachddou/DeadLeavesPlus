import numpy as np
import rasterio.features
from skimage.filters import gaussian
import shapely
from time import time



def sample_points_circle(n,radius):
    distance_to_center = np.random.randint(0,radius,n)
    angle = np.random.uniform(-np.pi,np.pi,n)
    
    x = np.array([radius+distance_to_center[i]*np.cos(angle[i]) for i in range(n)])
    y = np.array([radius+distance_to_center[i]*np.sin(angle[i]) for i in range(n)])
    return(np.stack([x,y],axis = -1))


def binary_polygon_generator(size, n = 100,concavity = 0.3,allow_holes = True, smoothing = True):
    t0 = time()
    coords = sample_points_circle(n,size//2)
    t1 = time()
    points = shapely.MultiPoint([(coords[k,0],coords[k,1]) for k in range(n)])
    t2 = time()
    concave_hull = shapely.concave_hull(points, ratio=concavity, allow_holes=allow_holes)
    t3 = time()
    img = rasterio.features.rasterize([concave_hull], out_shape=(size, size)).astype(np.bool_)
    t4 = time()
    
    # print(t1-t0)
    # print(t2-t1)
    # print(t3-t2)
    # print(t4-t3)
    if smoothing:
        min_blur = max(0.5,2*(size/200))
        max_blur = min(20,5*(size/200))
        blur_val = np.random.uniform(min_blur,max_blur)
        # blur_val = 10
        # print("The blur value is : "+str(blur_val))
        img = gaussian(img.astype(np.float32),blur_val)
        res = np.zeros(img.shape,dtype=np.bool_)
        res[img>0.5] = True
        
        t5 = time()
        # print(t5-t4)
        # print("final time:"+str(t5-t0))
        return(res)
    else:
        # print("final time:"+str(t4-t0))
        return(img)