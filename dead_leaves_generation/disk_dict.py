import numpy as np
from time import time
import os



def disk_dict(r_min,r_max):
    """
    function that creates a npy dictionnary of binary disk masks to speed up image generation.
    r_min : minimal radius
    r_max : maximal radius
    """
    disk_d = dict()
    for r in range(r_min,r_max+1,1):
        print(r)
        t0 = time()
        L = np.arange(-r,r + 1,dtype = np.int32)
        X, Y = np.meshgrid(L, L)
        disk_1d = np.array((X ** 2 + Y ** 2) <= r ** 2,dtype = bool)
        disk_d[str(r)] = disk_1d
        print(time()-t0)
    if not(os.path.isdir("npy")):
        os.makedirs("npy")
    np.save("npy/dict.npy", disk_d)


disk_dict(1,1000)
