#%%
import numpy as np
import skimage.io as skio
from skimage.filters import gaussian
import matplotlib.pyplot as plt
# %%
size  = 1000
imgBackGround = 255*np.ones((size,size,3))
maskBackGround = np.zeros((size,size))

posBackGroundDisk = [size//3,size//3]
posForeGroundDisk = [size//2,size//2]

colBackGroundDisk = np.array([255,0,0])
colForeGroundDisk = np.array([0,255,0])
L = np.arange(0,size,dtype = np.int32)
X, Y = np.meshgrid(L, L)

R1, R2 = size/4,size/4


# %%
maskBackGround = ((X-posBackGroundDisk[0])**2 + (Y-posBackGroundDisk[1])**2 < R1**2)
maskForeGround = ((X-posForeGroundDisk[0])**2 + (Y-posForeGroundDisk[1])**2 < R2**2)



##Blur_foreground


# %%
imgBackGround = 255*np.ones((size,size,3))
maskColor = np.repeat(np.expand_dims(maskBackGround,-1),3,axis=-1)
imgBackGround = np.uint8((1-maskColor)*imgBackGround+maskColor*colBackGroundDisk)


maskColorF = np.repeat(np.expand_dims(maskForeGround,-1),3,axis=-1)
imgForeGround = np.uint8((maskColorF*colForeGroundDisk))
# %%
plt.imshow(imgForeGround)

# %%
def fuse(backGround,sigma1,foreGround,foreMask,sigma2,gamma = 1/2.2):
    backGround = gaussian(backGround, sigma=sigma1)
    foreGround = gaussian(foreGround, sigma=sigma2)
    
    foreMask = np.repeat(np.expand_dims(foreMask,-1),3,axis = -1)
    #maskBlurred = gaussian(foreMask,sigma2+sigma1)**(1/gamma)
    maskBlurred = gaussian(foreMask,sigma2+sigma1)
    
    fused = (backGround**(1/gamma) *(1-maskBlurred) + foreGround**(1/gamma)*maskBlurred)**(gamma)
    return fused
# %%

res = fuse(imgBackGround,2,imgForeGround,maskForeGround,10,1/2.2)
plt.title("gamma = 2.2")
plt.imshow(res)
plt.savefig("gamma.png")
plt.show()
# %%
res = fuse(imgBackGround,2,imgForeGround,maskForeGround,10,1)
plt.title("gamma = 1")
plt.imshow(res)
plt.savefig("linear.png")

plt.show()
# %%
