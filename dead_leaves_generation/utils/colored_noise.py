import numpy as np

def sample_magnitude(resolution,slope):
    """function that creates a magnitude map for the frequency domain in 1/f^slope

    Args:
        resolution (int): size of the texture image
        slope (float): slope of the magnitude map in the log domain
    """
    fx, fy = np.meshgrid(range(resolution), range(resolution))

    fx = fx - resolution / 2
    fy = fy - resolution / 2
    
    fr = 1e-16 + np.abs(fx / resolution) ** slope +  np.abs(fy / resolution) ** slope

    magnitude = 1 / np.fft.fftshift(fr)
    magnitude[0,0] = 0
    magnitude = np.tile(magnitude[:,:,None], (1,1,3))
    return(magnitude)

def sample_color_noise(image,width,slope):
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