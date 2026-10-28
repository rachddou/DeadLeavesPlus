#%% imports

import matplotlib.pyplot as plt
import numpy as np
import shapely
import skimage.io as skio
import rasterio.features
#%%

def sample_points_circle(n,radius):
    """Generates n points sampled uniformly in a circle of radius radius.

    Args:
        n (int): number of points to generate.
        radius (int): radius of the circle.
    """
    distance_to_center = np.sqrt(np.random.uniform(0, radius**2, n))
    # distance_to_center = np.random.randint(0,radius,n)
    angle = np.random.uniform(-np.pi,np.pi,n)
    
    x = np.array([radius+distance_to_center[i]*np.cos(angle[i]) for i in range(n)])
    y = np.array([radius+distance_to_center[i]*np.sin(angle[i]) for i in range(n)])
    return(np.stack([x,y],axis = -1))

# %% sample points
n = 150
radius = 300
coords = sample_points_circle(n,radius)
points = shapely.MultiPoint([(coords[k,0],coords[k,1]) for k in range(n)])

points


# %%

from shapely.geometry import MultiPoint
from shapely.ops import triangulate
import matplotlib.pyplot as plt
import random

#%%
mp = MultiPoint(points)

# Delaunay triangulation with Shapely
tris =  triangulate(points)

# Plot
fig = plt.figure(frameon=False)
fig.set_size_inches(3,3)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

# # Plot triangles
# for tri in tris:
#     x, y = tri.exterior.xy
#     ax.plot(x, y, c = 'blue')

# Plot original points
px, py = zip(*coords)
ax.scatter(px, py)  
fig.savefig('delaunay_points.png', dpi = 300)
# plt.savefig('delaunay_points.png', dpi=300)
plt.show()


# %%



#%% 


for ratio in [0.1,0.3,0.4,0.5,0.8]:
    concave_hull = shapely.concave_hull(points, ratio=ratio,allow_holes=True)

    img = rasterio.features.rasterize([concave_hull], out_shape=(2*radius, 2*radius)).astype(np.bool_)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(3,3)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    for tri in tris:
        x, y = tri.exterior.xy
        ax.plot(x, y, c = 'blue')
    ax.imshow(1-img,cmap='gray',origin='lower')
    px, py = zip(*coords)
    ax.scatter(px, py)
    ax.axis('off')
    fig.savefig(f'delaunay_triangulation_{ratio}.png', dpi = 300)
    plt.show()


# %%
from skimage.filters import gaussian

concave_hull = shapely.concave_hull(points, ratio=0.4,allow_holes=True)

img = 1- rasterio.features.rasterize([concave_hull], out_shape=(2*radius, 2*radius)).astype(np.float32)
skio.imsave('original_shape.png', np.flipud(np.uint8(img*255)))
img = gaussian(img,10)
skio.imsave('blurred_shape.png', np.flipud(np.uint8(img*255)))

img[img>0.5] = 1
img[img<=0.5] = 0
skio.imsave('binarized_shape.png', np.flipud(np.uint8(img*255)))
# %%
