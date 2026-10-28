import numpy as np
import rasterio.features
import shapely
import math
import random
import cv2

from dead_leaves_generation.utils.kernels import fill_polygon, gaussian_threshold


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
    distance_to_center = np.random.randint(0,radius,n)
    angle = np.random.uniform(-np.pi,np.pi,n)

    x = radius + distance_to_center*np.cos(angle)
    y = radius + distance_to_center*np.sin(angle)
    return(np.stack([x,y],axis = -1))


def rasterize_polygon(geom, size):
    """Same output as ``rasterio.features.rasterize([geom], out_shape=(size, size))``
    for (multi)polygons, using a numba scanline fill instead of GDAL."""
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        coords = shapely.get_coordinates(geom)
        if geom.geom_type == "Polygon" and shapely.get_num_interior_rings(geom) == 0:
            ring_offsets = np.array([0, len(coords)], dtype=np.int64)
        else:
            ring_sizes = shapely.get_num_coordinates(shapely.get_rings(geom))
            ring_offsets = np.concatenate([[0], np.cumsum(ring_sizes)]).astype(np.int64)
    else:
        # degenerate hulls (points / lines) keep GDAL's line burning rules
        return rasterio.features.rasterize([geom], out_shape=(size, size)).astype(np.bool_)
    return fill_polygon(np.ascontiguousarray(coords[:, 0]), np.ascontiguousarray(coords[:, 1]),
                        ring_offsets, size, size)


def _gaussian_kernel(sigma, truncate=4.0):
    """1D kernel used by scipy.ndimage.gaussian_filter (and thus skimage's gaussian)."""
    radius = int(truncate * float(sigma) + 0.5)
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / (sigma * sigma) * x ** 2)
    return phi_x / phi_x.sum()


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

    points = shapely.multipoints(coords)

    concave_hull = shapely.concave_hull(points, ratio=concavity, allow_holes=allow_holes)

    img = rasterize_polygon(concave_hull, size)

    if smoothing:
        min_blur = max(0.5,2*(size/200))
        max_blur = min(10,5*(size/200))
        blur_val = np.random.uniform(min_blur,max_blur)
        # skimage.filters.gaussian(img.astype(np.float32), blur_val) > 0.5
        return gaussian_threshold(img, _gaussian_kernel(blur_val), 0.5)
    else:
        return(img)
