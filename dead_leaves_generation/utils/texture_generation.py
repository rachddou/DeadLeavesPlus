import numpy as np
from skimage.color import rgb2lab, lab2rgb
from dead_leaves_generation.utils.interpolation_maps import sample_grid,  sample_sinusoid,sample_interpolation_map
from dead_leaves_generation.utils.geometric_perturbation import generate_perturbation

from dead_leaves_generation.utils.colored_noise import sample_color_noise



## a bilevel texture function that mixes either two colors or two micro-texture maps
def bilevelTextureMixer(color_source_1 = np.random.randint(0,255,(100,100,3)),color_source_2 = np.random.randint(0,255,(100,100,3)),single_color1 = True,single_color2 = True,mixing_types = ["sin"],width = 1000,thresh_val = 10,warp = True):
    """function that mixes two color/texture maps with either a sinusoidal pattern, a grid pattern or a noise pattern.

    Args:
        color_source_1 (_type_, optional): source color image 1. Defaults to np.random.randint(0,255,(100,100,3)).
        color_source_2 (_type_, optional): source color image 2. Defaults to np.random.randint(0,255,(100,100,3)).
        single_color1 (bool, optional): whether texture 1 is a single color or a colored noise map. Defaults to True.
        single_color2 (bool, optional): whether texture 2 is a single color or a colored noise map. Defaults to True.
        mixing_types (list, optional): interpolation methods between two textures. Defaults to ["sin"].
        width (int, optional): size of the final texture image. Defaults to 1000.
        thresh_val (int, optional): specific to noise mixing, the thresholding parameter for binary masking. Defaults to 10.
        warp (bool, optional): whether or not to apply warping for the sinusoidal interpolation maps. Defaults to True.
    """
    if single_color1:
        texture_map_1 = color_source_1[np.random.randint(0,color_source_1.shape[0],1),np.random.randint(0,color_source_1.shape[1],1),:].reshape((1,1,3))

    else:
        slope1 = np.random.uniform(0.5,2.5)
        texture_map_1 = sample_color_noise(color_source_1,width,slope1)
    if single_color2:
        texture_map_2 = color_source_2[np.random.randint(0,color_source_2.shape[0],1),np.random.randint(0,color_source_2.shape[1],1),:].reshape((1,1,3))

    else:
        #ad hoc ok
        slope2 = np.random.uniform(0.5,2.5)
        texture_map_2 =sample_color_noise(color_source_2,width,slope2)

    interpolation_map = sample_interpolation_map(mixing_types = mixing_types,width = width,thresh_val = thresh_val,warp = warp)
    texture_map_1 = texture_map_1/255.
    texture_map_2 = texture_map_2/255.
    texture_transform_1 = rgb2lab(texture_map_1)
    texture_transform_2 = rgb2lab(texture_map_2)
    r = interpolation_map*(texture_transform_2[...,0]-texture_transform_1[...,0])+texture_transform_1[...,0]
    g = interpolation_map*(texture_transform_2[...,1]-texture_transform_1[...,1])+texture_transform_1[...,1]
    b = interpolation_map*(texture_transform_2[...,2]-texture_transform_1[...,2])+texture_transform_1[...,2]

    res_image = np.stack([r,g,b],axis = -1)

    res_image = lab2rgb(res_image)

    res_image = np.uint8(res_image*255)
    return(res_image)


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
    
    res_image = np.zeros((width,width))

    color_1 = color_1/255.
    color_2 = color_2/255.
    texture_transform_1 = rgb2lab(color_1)
    texture_transform_2 = rgb2lab(color_2)
    
    pattern = np.zeros((width,width))

    if type == "sin":
        pattern = sample_sinusoid(width = width,angle = angle,variable_freq=(np.random.random()>0.5))
    elif type == "grid":
        pattern = sample_grid(width = width,period = period,angle = angle ,thickness = thickness)


    if warp:        
        pattern = np.clip(generate_perturbation(pattern),0,1)  


    r = pattern*(texture_transform_2[...,0]-texture_transform_1[...,0])+texture_transform_1[...,0]
    g = pattern*(texture_transform_2[...,1]-texture_transform_1[...,1])+texture_transform_1[...,1]
    b = pattern*(texture_transform_2[...,2]-texture_transform_1[...,2])+texture_transform_1[...,2]

    res_image = np.stack([r,g,b],axis = -1)

    res_image = lab2rgb(res_image)

    res_image = np.uint8(res_image*255)

    return(res_image)