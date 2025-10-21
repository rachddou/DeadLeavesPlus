import os
import argparse
import numpy as np 
import skimage.io as skio
from multiprocessing import Pool
from time import time, clock_gettime, CLOCK_MONOTONIC, sleep
import hydra
from skimage.color import rgb2gray
from dead_leaves_generation.dead_leaves_class import Textures, Deadleaves 

dict_instance = np.load('npy/dict.npy',allow_pickle=True)


def image_gen(object,config):
    """a function that generates either a dead leaves or a texture image, and saves it in a directory, based on the configuration file

    Args:
        object (_type_): instance of the class that generates the image
        config (_type_): config file that contains the parameters of the image generation
    """
    if config.image_type  == "dead_leaves":
        if config.shape.multiple_shapes:
            shapes = ["poly","mix"]
            shape_type = np.random.choice(shapes,1,p = [0.5,0.5])
            object.shape_type = shape_type
        if config.post_process.dof:
            sigma_depth = 5*np.random.power(1/2)
        else:
            sigma_depth = 0
        object.update_leaves_size_parameters(config.shape.rmin,config.shape.rmax)
        if config.test:
            object.source_image_sampling()
            object.generate_stack(15000)
        else:
            fetch = (config.texture.texture_path != "")
            object.compose_dead_leaves_depth_of_field(sigma_depth,fetch = fetch)
        object.postprocess(blur = config.post_process.blur,ds = config.post_process.downscaling)
    elif config.image_type  == "textures":
        object.source_image_sampling()
        object.resulting_image = object.generate_texture(width = config.size, angle = np.random.randint(0,180))
    # saving the image
    new_dir = os.path.join(config.io.path_origin,config.io.path)
    filename = '{}/im_{}.png'.format(new_dir,int(100*clock_gettime(CLOCK_MONOTONIC)))
    skio.imsave(filename, np.uint8(object.resulting_image))
    object.clear()
    return(object)


@hydra.main(version_base=None, config_path="./config/generated_config/", config_name="default")
def main(config):
    """
    Main function that generates the images based on the configuration file
    This function creates a directory, generates the images, and saves them in the directory
    It can be passed as an argument to the hydra.main function, which will use the configuration file to generate the images
    Args:
        config (_type_): configuration file that contains the parameters of the image generation
    """
    # directory creation
    direct = config.io.path
    if not os.path.exists(os.path.join(config.io.path_origin,direct)):
        os.makedirs(os.path.join(config.io.path_origin,direct))
    new_dir = os.path.join(config.io.path_origin,direct)

    # object creation
    if config.image_type  == "dead_leaves":
        object  = Deadleaves(rmin = config.shape.rmin,rmax = config.shape.rmax,alpha = config.shape.alpha,width = config.size,\
                            natural = config.color.natural, path = config.color.color_path,\
                            shape_type = config.shape.shape_type,texture_types=config.texture.texture_types,\
                            texture_type_frequency=config.texture.texture_type_frequency, slope_range = config.texture.slope_range,\
                            texture = config.texture.texture,gen = config.texture.texture_gen,warp = config.texture.warp,\
                            rdm_phase = config.texture.rdm_phase,perspective=config.texture.perspective,\
                            texture_path= config.texture.texture_path)
    elif config.image_type  == "textures":
        object = Textures(width = config.size,natural = config.color.natural, path = config.color.color_path, texture_types=config.texture.texture_types,texture_type_frequency=config.texture.texture_type_frequency,\
                          slope_range = config.texture.slope_range,warp = config.texture.warp)
    else:
        raise Exception("wrong image type name")
    for _ in range(config.number):
        # changing shape types 
        t0 = time()
        # composing our image with depth
        if config.image_type  == "dead_leaves":
            if config.shape.multiple_shapes:
                shapes = ["poly","mix"]
                shape_type = np.random.choice(shapes,1,p = [0.5,0.5])
                object.shape_type = shape_type
            blur_val = 0
            if config.post_process.dof:
                if config.post_process.blur_type == "gaussian":
                    blur_val = 5*np.random.power(1/2)
                elif config.post_process.blur_type == "lens":
                    blur_val = 10*np.random.power(1/2)
            object.update_leaves_size_parameters(config.shape.rmin,config.shape.rmax)
            if config.test:
                object.source_image_sampling()
                object.generate_stack(15000)
            else:
                fetch = (config.texture.texture_path != "")
                object.compose_dead_leaves_depth_of_field(blur_type = config.post_process.blur_type,blur_val=blur_val,fetch = fetch)
            object.postprocess(blur = config.post_process.blur,ds = config.post_process.downscaling)

        elif config.image_type  == "textures":
            object.source_image_sampling()
            object.resulting_image = object.generate_texture(width = config.size, angle = np.random.randint(0,180))

        # saving the image
        filename = '{}/im_{}.png'.format(new_dir,int(100*clock_gettime(CLOCK_MONOTONIC)))
        if config.color.grey:
            skio.imsave(filename, np.uint8(np.clip(255*rgb2gray(object.resulting_image),0,255)))
        else:
            skio.imsave(filename, np.uint8(object.resulting_image))
        object.clear()
        print(time()-t0)

if __name__ == """__main__""":
    main()


