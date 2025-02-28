import math, random
from typing import List, Tuple
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from time import time
import numpy as np
from numba import njit
import cv2



def rotate_image(img, angle):
    size_reverse = np.array(img.shape[1::-1]) # swap x with y
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.)
    MM = np.absolute(M[:,:2])
    size_new = MM @ size_reverse
    M[:,-1] += (size_new - size_reverse) / 2.
    return cv2.warpAffine(img, M, tuple(size_new.astype(int)))

# def rotate_CV(image, angle ):
#     #in OpenCV we need to form the tranformation matrix and apply affine calculations
#     #
#     h,w = image.shape[:2]
#     cX,cY = (w//2,h//2)
#     M = cv2.getRotationMatrix2D((cX,cY),angle,1)
#     rotated = cv2.warpAffine(image,M , (w,h),flags=cv2.INTER_LINEAR)
#     return rotated



def make_rectangle_mask(radius):
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


def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int) -> List[Tuple[float, float]]:
    """
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irregularity (float):
            variance of the spacing of the angles between consecutive
            vertices.
        spikiness (float):
            variance of the distance of each vertex to the center of
            the circumference.
        num_vertices (int):
            the number of vertices of the polygon.
    Returns:
        List[Tuple[float, float]]: list of vertices, in CCW order.
    """
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)

    # now generate the points
    points = []
    points = np.zeros((num_vertices,2))
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        # point = (center[0] + radius * math.cos(angle),
        #          center[1] + radius * math.sin(angle))
        # points.append(point)
        points[i,0] = center[0] + radius * math.cos(angle)
        points[i,1] = center[1] + radius * math.sin(angle)
        angle += angle_steps[i]

    return points


def random_angle_steps(steps: int, irregularity: float) -> List[float]:
    """Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    # generate n angle steps
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles

def clip(value, lower, upper):
    """
    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))

@njit
def trace(vert1,vert2,boundary):
    xs = [vert1[0],vert2[0]]
    ys  = [vert1[1],vert2[1]]
    ny = abs(ys[1]-ys[0])
    nx = abs(xs[1]-xs[0])
    if  nx > ny:
        YY,XX = interpolate(ys,xs)
    else :
        XX,YY = interpolate(xs,ys)
    for i in range(XX.shape[0]):
        boundary[XX[i],YY[i]] +=True
#     boundary[XX,YY] += True
    return(boundary)

@njit
def create_boundary_from_vertices(vertices):
    x_max = vertices[:,0].max()
    y_max = vertices[:,1].max()
    boundary = np.zeros((x_max+1+x_max%2,y_max+1+y_max%2),dtype = np.bool_)
    n = vertices.shape[0]
    for i in range(n-1):
        boundary = trace(vertices[i],vertices[i+1],boundary)
    boundary =  trace(vertices[-1],vertices[0],boundary)
    return(boundary)

@njit
def interpolate(x,y):
    slope = (x[1]-x[0])/(abs(y[1]-y[0])+1)
    YY = np.arange(min(y),max(y)+1,1)
    if y[0] > y[1]:
        YY = np.flip(YY)
    XX = np.array([int(round(x[0]+slope*i,0)) for i in range(YY.size)])
    
    return(XX,YY)

@njit
def create_mask_from_boundary_1d(boundary):
    H,W = boundary.shape[0],boundary.shape[1]
    mask1 = np.zeros((H,W),dtype = np.bool_)
    mask1+=boundary
    for i in range(1,H):
        for j  in range(1,W):
                if mask1[i-1,j] == True and mask1[i,j-1] == True:
                    mask1[i,j] += 1
    return(mask1)

@njit
def create_mask_from_boundary(boundary):
    mask1 = create_mask_from_boundary_1d(boundary)
    mask2 = np.flipud(create_mask_from_boundary_1d(np.flipud(boundary)))
    mask3 = np.fliplr(create_mask_from_boundary_1d(np.fliplr(boundary)))
    mask = mask1*mask2*mask3
    return(mask)


def create_mask(center,avg_radius,irregularity,spikiness,num_vertices):
    t0 = time()
    vertices = generate_polygon(center,avg_radius,irregularity,spikiness,num_vertices)
    t1 = time()
    vertices[:,0] =  vertices[:,0]-vertices[:,0].min()
    vertices[:,1] =  vertices[:,1]-vertices[:,1].min()
    vertices = np.int16(vertices)
    t2 = time()
    boundary = create_boundary_from_vertices(vertices)
    t3 = time()
    mask = create_mask_from_boundary(boundary)
    t4 = time()
    return(mask)


if __name__ == "__main__":


    def generate_polygon(center: Tuple[float, float], avg_radius: float,
                        irregularity: float, spikiness: float,
                        num_vertices: int) -> List[Tuple[float, float]]:
        """
        Start with the center of the polygon at center, then creates the
        polygon by sampling points on a circle around the center.
        Random noise is added by varying the angular spacing between
        sequential points, and by varying the radial distance of each
        point from the centre.

        Args:
            center (Tuple[float, float]):
                a pair representing the center of the circumference used
                to generate the polygon.
            avg_radius (float):
                the average radius (distance of each generated vertex to
                the center of the circumference) used to generate points
                with a normal distribution.
            irregularity (float):
                variance of the spacing of the angles between consecutive
                vertices.
            spikiness (float):
                variance of the distance of each vertex to the center of
                the circumference.
            num_vertices (int):
                the number of vertices of the polygon.
        Returns:
            List[Tuple[float, float]]: list of vertices, in CCW order.
        """
        # Parameter check
        if irregularity < 0 or irregularity > 1:
            raise ValueError("Irregularity must be between 0 and 1.")
        if spikiness < 0 or spikiness > 1:
            raise ValueError("Spikiness must be between 0 and 1.")

        irregularity *= 2 * math.pi / num_vertices
        spikiness *= avg_radius
        angle_steps = random_angle_steps(num_vertices, irregularity)

        # now generate the points
        points = []
        angle = random.uniform(0, 2 * math.pi)
        for i in range(num_vertices):
            radius = clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
            point = (center[0] + radius * math.cos(angle),
                    center[1] + radius * math.sin(angle))
            points.append(point)
            angle += angle_steps[i]

        return points
    def random_angle_steps(steps: int, irregularity: float) -> List[float]:
        """Generates the division of a circumference in random angles.

        Args:
            steps (int):
                the number of angles to generate.
            irregularity (float):
                variance of the spacing of the angles between consecutive vertices.
        Returns:
            List[float]: the list of the random angles.
        """
        # generate n angle steps
        angles = []
        lower = (2 * math.pi / steps) - irregularity
        upper = (2 * math.pi / steps) + irregularity
        cumsum = 0
        for i in range(steps):
            angle = random.uniform(lower, upper)
            angles.append(angle)
            cumsum += angle

        # normalize the steps so that point 0 and point n+1 are the same
        cumsum /= (2 * math.pi)
        for i in range(steps):
            angles[i] /= cumsum
        return angles
    for spk in range(10):
        for n in range(10):
            vertices = generate_polygon(center=(250, 250),
                                    avg_radius=100,
                                    irregularity=0.35,
                                    spikiness=0.05*spk+0.1,
                                    num_vertices=10+n*5)

            black = (0, 0, 0)
            white = (255, 255, 255)
            img = Image.new('RGB', (500, 500), white)
            im_px_access = img.load()
            draw = ImageDraw.Draw(img)

            # either use .polygon(), if you want to fill the area with a solid colour
            draw.polygon(vertices, outline=black, fill=white)

            # or .line() if you want to control the line thickness, or use both methods together!
            draw.line(vertices + [vertices[0]], width=2, fill=black)
            img.save("polygons/polygons_n{}_spk{}.png".format(str(10+n*5),str(int(100*(0.05*spk+0.1))/100.)))


    # now you can save the image (img), or do whatever else you want with it.