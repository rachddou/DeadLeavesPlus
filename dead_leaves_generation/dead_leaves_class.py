import os
import numpy as np 
import skimage.io as skio
import matplotlib.pyplot as plt
import random
from time import time
import cv2
import random
from blurgenerator import lens_blur


from dead_leaves_generation.utils import theoretical_number_disks,linear_color_gradient,pattern_patch_two_colors,random_phase_im,freq_noise,mixing_materials_v2,perspective_shift
from dead_leaves_generation.polygons_maker_bis import binary_polygon_generator, make_rectangle_mask


dict_instance = np.load('npy/dict.npy',allow_pickle=True)

class Textures:
    def __init__(self,width = 1000,natural = True, path = "", texture_types = ["sin"],texture_type_frequency = [1],slope_range = [0.5,2.5],img_source = np.random.randint(0,255,(1000,1000,3)),warp = True,rdm_phase = False):

        self.width = width
        self.natural = natural
        self.path = path
        self.warp = warp
        self.rdm_phase = rdm_phase
        self.gen = True

        self.img_source = img_source
        self.w = img_source.shape[0]
        self.l = img_source.shape[1]
        self.files = []
        self.resulting_image = np.ones((width,width,3), dtype = np.uint8)
        self.perspective_shift = False
        self.perspective_var = True
        
        if self.natural:
            self.source_image_sampling()
        self.texture_type_lists = texture_types
        self.texture_type_frequency = texture_type_frequency
        self.slope_range = slope_range
        self.texture_type = texture_types[0]
        self.sample_texture_type()

    def lin_gradient(self,color1,color2,angle = 45):
        k = np.random.uniform(0.1,0.5)
        color_component = linear_color_gradient(color_1 = color1,color_2 = color2,width=self.width,angle = angle, k = k,color_space = "lab")
        return(color_component)
    
    def random_patch_selection(self):
        x = np.random.randint(0,max(1,self.w - 101))
        y = np.random.randint(0,max(1,self.l - 101))
        random_patch = self.img_source[x:x+100,y:y+100]
        return(random_patch)
    def generate_source_texture(self,width = 1000, angle = 45,single_color = True):


        if  self.texture_type == "freq_noise":
            slope = np.random.uniform(self.slope_range[0],self.slope_range[1])
            #color_component = freq_noise(self.img_source,width=self.width,slope = slope)
            color_component = freq_noise(self.random_patch_selection(),width=width,slope = slope)
            
        elif self.texture_type == "texture_mixes":
            single1 = np.random.random()<0.1
            single2 = np.random.random()<0.1
            if self.warp:
                warp = np.random.random()<0.5
            else:
                warp = False
            s = np.random.uniform(10,20)
            t = np.random.uniform(s//2,s)
            thresh = np.random.randint(5,50)
            poss = [["sin"],["grid"]]
            mixing_type  = np.random.choice(np.arange(0,2,1),p = np.array([0.9,0.1])) 
            color_component = mixing_materials_v2(tmp1 = self.random_patch_selection(),tmp2 = self.random_patch_selection(),single_color1=single1,\
                                                  single_color2=single2,mixing_types=poss[mixing_type],\
                                                  width = width,thresh_val = thresh,warp = warp)
        
        else:
            t_min,t_max= 20,200
            if self.perspective_shift:
                self.perspective_var = np.random.random()>0.5

            color = np.uint8(self.img_source[np.random.randint(0,self.w),np.random.randint(0,self.l),:])
            color_2 = np.uint8(self.img_source[np.random.randint(0,self.w),np.random.randint(0,self.l),:])
            thickness =  random.randint(1,3)
            if self.warp:
                warp_var = np.random.random()<0.5
            else:
                warp_var = False
            color_component = pattern_patch_two_colors(color, color_2,width=width,angle = angle,thickness = thickness,warp = warp_var, type = self.texture_type)
        return(color_component)
    
    def source_image_sampling(self):
        if not(self.files):
            self.files = [os.path.join(self.path,f) for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]
        N = len(self.files)
        ind = random.randint(0,N-1)
        self.img_source = skio.imread(self.files[ind])
        self.w = self.img_source.shape[0]
        self.l = self.img_source.shape[1]
        print(self.files[ind])
        
    def sample_texture_type(self):
        if len(self.texture_type_lists) == 1:
            self.texture_type = self.texture_type_lists[0]
        else:
            self.texture_type = np.random.choice(self.texture_type_lists,p = self.texture_type_frequency)
            
    def generate_texture(self,width = 100,angle = 45):
        t0 = time()
        self.sample_texture_type()
        tmp_rdm_phase = self.rdm_phase
        if not(self.gen):
            print(self.texture_type)
        if self.texture_type == "gradient":
            color1 = np.uint8(self.img_source[np.random.randint(0,self.w),np.random.randint(0,self.l),:])
            color2 = np.uint8(self.img_source[np.random.randint(0,self.w),np.random.randint(0,self.l),:])
            res = self.lin_gradient(color1,color2,angle = angle)
        else:
            
            if self.texture_type in ["grid","texture_mixes"] and self.rdm_phase:
                self.rdm_phase = False
            single_color = np.random.random()>0.5
            self.perspective_var = np.random.random()>0.5
            if self.perspective_var and self.perspective_shift:
                res = self.generate_source_texture(width = 2*width, angle = angle,single_color = single_color)
                print("perspective shifting")
                res = perspective_shift(res)
            else:
                res = self.generate_source_texture(width = width, angle = angle,single_color = single_color)
            self.rdm_phase = tmp_rdm_phase
        return(res)


    def clear(self):
        self.resulting_image = np.ones((self.width,self.width,3), dtype = np.uint8)


class Deadleaves(Textures):
    def __init__(self,rmin = 1,rmax = 1000,alpha = 3,width = 1000,natural = True, path = "",texture_path = "", shape_type = "poly",texture_types = ["sin"],texture_type_frequency = [1],slope_range = [0.5,2.5], texture = True,gen = False,warp = True,rdm_phase = False, perspective = True, img_source = np.random.randint(0,255,(1000,1000,3))):
        super(Textures).__init__()
        self.rmin = rmin
        self.rmax = rmax
        self.alpha = alpha
        self.width = width
        self.natural = natural
        self.path = path
        self.warp = warp
        self.rdm_phase = rdm_phase
        self.texture_path = texture_path
        self.shape_type = shape_type
        self.texture = texture
        self.texture_type_lists = texture_types
        self.texture_type_frequency = texture_type_frequency
        self.slope_range = slope_range
        self.texture_type = self.sample_texture_type()
        self.perspective_shift = perspective
        self.perspective_var = True

        self.files = []


        self.img_source = img_source

        self.w = self.img_source.shape[0]
        self.l = self.img_source.shape[1]
        
        self.gen = gen
        self.n_textures = 10

        self.textures   = []

        self.binary_image = np.ones((width,width), dtype = bool)
        self.resulting_image = np.ones((width,width,3), dtype = np.uint8)

        self.vamin = 1/(rmax**(alpha-1))
        self.vamax = 1/(rmin**(alpha-1))
        theoretical_n_shapes = theoretical_number_disks(0.002,rmin,rmax,width,alpha)
        interval = int(theoretical_n_shapes)
        print(interval)
        if interval <10:
            interval = 1
        if interval >10:
            interval = interval//10
        self.interval = interval


    def update_leaves_size_parameters(self,rmin,rmax):
        self.rmin = rmin
        self.rmax = rmax
        self.vamin = 1/(self.rmax**(self.alpha-1))
        self.vamax = 1/(self.rmin**(self.alpha-1))
        theoretical_n_shapes = theoretical_number_disks(0.002,self.rmin,self.rmax,self.width,self.alpha)
        interval = int(theoretical_n_shapes)
        print(interval)
        if interval <10:
            interval = 1
        if interval >10:
            interval = interval//10
        self.interval = interval

        
    def fetch_textures(self):
        files = [os.path.join(self.texture_path,f) for f in os.listdir(self.texture_path)]
        file_id = np.random.choice(len(files),self.n_textures)
        textures = [skio.imread(files[ind])for ind in file_id]
        self.textures =  textures
        return(textures)
    def random_patch_selection(self):
        #check
        x = np.random.randint(0,max(1,self.w - 101))
        y = np.random.randint(0,max(1,self.l - 101))
        random_patch = self.img_source[x:x+100,y:y+100,:]
        return(random_patch)
    
    def generate_textures_dictionary(self):
        
        textures = []
        for _ in range(self.n_textures):
            angle = np.random.randint(0,360)
            texture_img = self.generate_texture(width = 2*self.rmax+100,angle = angle)
            scale_dict = {"1":texture_img}
            
            # this process takes 5 seconds on a normal cpu
            for i in range(2,6):
                # scale_dict[str(i)] = np.uint8(255*pyramid_reduce(texture_img,downscale = i ,channel_axis = 2))
                scale_dict[str(i)] = cv2.resize(texture_img,(0,0),fx = 1./i,fy = 1./i,interpolation = cv2.INTER_AREA)
            #print(time()-t1)
            textures.append(scale_dict)
        self.textures =  textures

    def pick_texture(self,size):
        current_texture_dict = self.textures[np.random.randint(0,self.n_textures)]

        h = current_texture_dict["1"].shape[0]
        max_scale = min(5,h/size)
        scale = np.floor(1 + max(0,(max_scale-1)*np.random.power(2/3))).astype(int)
        current_texture = current_texture_dict[str(scale)]

        transform = np.random.choice(4)
        if transform == 0:
            current_texture = np.flipud(current_texture)
        if transform == 1:
            current_texture = np.fliplr(current_texture)
        if transform == 2:
            current_texture = np.rot90(current_texture,axes = (0,1))
        if transform == 3:
            current_texture = np.flipud(np.rot90(current_texture,axes = (0,1)))
        return(current_texture)
    
    def resize_textures(self,size,texture):
        h = texture.shape[0]
        max_scale = min(5,h/size)
        scale = 1 +  (max_scale-1)*np.random.power(2/3)

        scale = int(scale*5)/5.
        if scale > 1:

            # texture_resized = pyramid_reduce(texture,downscale = scale,channel_axis = 2)
            texture_resized = cv2.resize(texture,(0,0),fx = 1./scale,fy = 1./scale,interpolation = cv2.INTER_AREA)
            # texture_resized = np.uint8(255*texture_resized)
        else :
            texture_resized = texture
        
        return(texture_resized)

    def generate_single_shape_mask(self):
        radius = self.vamin + (self.vamax-self.vamin)*np.random.random()
        radius = int(1/(radius**(1./(self.alpha-1))))
        shape = self.shape_type
        if shape == "mix":
            shape = random.choice(["disk","poly","rectangle"])
        if shape == "poly":
            shape_1d = binary_polygon_generator(2*(3*radius//2)+1,n= np.random.randint(50,max(100,0.9*radius)), allow_holes=bool(random.getrandbits(1)),smoothing=bool(random.getrandbits(1)))
            
            if max(shape_1d.shape[0],shape_1d.shape[1]) >=2*self.rmax+100:
                scale = (2*self.rmax+100.)/max(shape_1d.shape[0],shape_1d.shape[1])
                new_size = (int(2*((shape_1d.shape[0]*scale)//2)-1),int(2*((shape_1d.shape[1]*scale)//2)-1))
                shape_1d = np.bool_(cv2.resize(np.uint8(shape_1d),new_size, interpolation = cv2.INTER_AREA))
        elif shape == "disk":
            shape_1d = dict_instance[()][str(radius)]
        elif shape == "rectangle":
            shape_1d = make_rectangle_mask(radius)
            if max(shape_1d.shape[0],shape_1d.shape[1]) >=2*self.rmax+100:
                scale = (2*self.rmax+100.)/max(shape_1d.shape[0],shape_1d.shape[1])
                new_size = (int(2*((shape_1d.shape[0]*scale)//2)-1),int(2*((shape_1d.shape[1]*scale)//2)-1))
                shape_1d = np.bool_(cv2.resize(np.uint8(shape_1d),new_size, interpolation = cv2.INTER_AREA))
        return(shape_1d,radius)
    
    def add_shape_to_binary_mask(self,shape_1d):
        width_shape,length_shape = shape_1d.shape[0],shape_1d.shape[1]
        pos = [np.random.randint(0,self.width),np.random.randint(0,self.width)]
        
        # defining useful positions
        x_min = max(0,pos[0]-width_shape//2)
        x_max = min(self.width,1+pos[0]+width_shape//2)
        y_min = max(0,pos[1]-length_shape//2)
        y_max = min(self.width,1+pos[1]+length_shape//2)

        shape_mask_1d = self.binary_image[x_min:x_max,y_min:y_max].copy()
        shape_1d = shape_1d[max(0,width_shape//2-pos[0]):min(width_shape,self.width+width_shape//2-pos[0]),max(0,length_shape//2-pos[1]):min(length_shape,self.width+length_shape//2-pos[1])]

        shape_mask_1d *=  shape_1d
        self.binary_image[x_min:x_max,y_min:y_max]*=np.logical_not(shape_mask_1d)
        return(x_min,x_max,y_min,y_max,shape_mask_1d)
    
    def render_shape(self,shape_mask_1d,radius):
        t = time()
        width_shape,length_shape = shape_mask_1d.shape[0],shape_mask_1d.shape[1]

        shape_mask= np.float32(np.repeat(shape_mask_1d[:, :, np.newaxis], 3, axis=2))
        shape_render = shape_mask.copy()
        color = np.uint8(self.img_source[np.random.randint(0,self.w),np.random.randint(0,self.l),:])
        if radius<20:
            shape_render = color*shape_render
        else:
            angle = np.random.randint(0,360)
            color_2 = np.uint8(self.img_source[np.random.randint(0,self.w),np.random.randint(0,self.l),:])
            grad_vs_texture = np.random.random()
            if not(self.texture):
                grad_vs_texture = 1
            if grad_vs_texture > 0.95:
                 # linear gradient
                k = np.random.uniform(0.1,0.5)
                color_component = linear_color_gradient(color,color_2,width=2*max(width_shape,length_shape)+1,angle = angle, k = k,color_space = "lab")

            else:
                if self.gen:
                    color_component = self.generate_texture(width=60+max(width_shape,length_shape),angle = angle)
                else:
                    color_component = self.pick_texture(size = max(width_shape,length_shape))


            h,w = color_component.shape[0],color_component.shape[1]
            x,y = np.random.randint(0,max(1,h - width_shape)), np.random.randint(0,max(1,w - length_shape))
            color_component = color_component[x:x+width_shape,y:y+length_shape]
            # color_component = color_component[np.floor(h//2-width_shape/2).astype(np.uint16):np.floor(h//2+width_shape/2).astype(np.uint16),np.floor(w//2-length_shape/2).astype(np.uint16):np.floor(w//2+length_shape/2).astype(np.uint16),:]
            shape_render= np.uint8(np.float32(shape_render)*color_component)
        
        return(shape_mask,shape_render)

    def generate_stack(self,disk_count):

        for i in range(disk_count):
            shape_1d,radius = self.generate_single_shape_mask()
            x_min,x_max,y_min,y_max,shape_mask_1d = self.add_shape_to_binary_mask(shape_1d)
            shape_mask,shape_render = self.render_shape(shape_mask_1d,radius)
            self.resulting_image[x_min:x_max,y_min:y_max,:]*=np.uint8(1-shape_mask)
            self.resulting_image[x_min:x_max,y_min:y_max,:]+=np.uint8(shape_render)
        print("dead_leaves stack created")
        return(self.resulting_image.copy(),np.uint8(1-np.repeat(self.binary_image[...,np.newaxis],3,axis= 2)).copy())
    
    def clear(self):
        self.resulting_image = np.ones((self.width,self.width,3), dtype = np.uint8)
        self.binary_image = np.ones((self.width,self.width), dtype = bool)

    def source_image_sampling(self):
        if not(self.files):
            self.files = [os.path.join(self.path,f) for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]
        N = len(self.files)
        ind = random.randint(0,N-1)
        self.img_source = skio.imread(self.files[ind])
        self.w = self.img_source.shape[0]
        self.l = self.img_source.shape[1]
        print(self.files[ind])

    def compose_dead_leaves_depth_of_field(self,blur_type,blur_val,fetch = False):
        
        if self.natural:
            self.source_image_sampling()
            if not(self.img_source.shape[2] ==3):
                self.source_image_sampling()
        if self.texture:
            if not(self.gen):
                if fetch:
                    self.fetch_textures()
                else:
                    self.generate_textures_dictionary()

        background,back_mask = self.generate_stack(10*self.interval)
        self.clear()
        plainground,plain_mask = self.generate_stack(self.interval)
        self.clear()

        foreground,fore_mask = self.generate_stack(int(0.5*self.interval))

        # adding the in focus plain ground to the out of focus background
        if blur_type == "gaussian":
            background = cv2.GaussianBlur(background,(11,11),sigmaX = blur_val, borderType = cv2.BORDER_DEFAULT)
        elif blur_type == "lens":
            background = lens_blur(background, radius=blur_val, components=2, exposure_gamma=2)
        plainground =(1-plain_mask)*background + plain_mask*plainground

        # adding the out of focus foreground to the in-focus background

        foreground+=(1-fore_mask)*plainground
        if blur_type == "gaussian":
            foreground = cv2.GaussianBlur(foreground,(11,11),sigmaX = blur_val, borderType = cv2.BORDER_DEFAULT)
            fore_mask = cv2.GaussianBlur(fore_mask,(11,11),sigmaX = blur_val, borderType = cv2.BORDER_DEFAULT)
        elif blur_type == "lens":
            foreground = lens_blur(foreground, radius=blur_val, components=2, exposure_gamma=2)
            fore_mask = lens_blur(255*fore_mask, radius=blur_val, components=2, exposure_gamma=2)/255.
        
        im2 = plainground*(1-fore_mask)
        res=foreground*fore_mask
        im2+=res
        self.resulting_image = np.clip(im2,0,255)

    def postprocess(self,blur=True,ds=True):
        if blur or ds:
            if blur:
                blur_value = np.random.uniform(1,3)
                self.resulting_image = cv2.GaussianBlur(self.resulting_image,(11,11),sigmaX =  blur_value, borderType = cv2.BORDER_DEFAULT)
            if ds:
                self.resulting_image = cv2.resize(self.resulting_image,(0,0), fx = 1/2.,fy = 1/2. , interpolation = cv2.INTER_AREA)
            self.resulting_image = np.uint8(self.resulting_image)


            


if __name__ == "__main__":
    object = Deadleaves(rmin = 20,rmax = 400,alpha = 3,
                                width = 1000,natural = True, path = "/Users/raphael/Workspace/telecom/code/ffdnet_core/datasets/HIGH X2 Urban",
                                shape_type = "mix", texture = True)



    yo = object.random_patch_selection()




    shapes = ["poly","disk","rectangle","mix"]
    shape_type = "mix"
   #shape_type = np.random.choice(shapes,1,p = [0.2,0.2,0.2,0.4])
    object.shape_type = shape_type

    sigma_depth = 5*np.random.power(1/2)
    # object.update_leaves_size_parameters(20,400)
    object.compose_dead_leaves_depth_of_field(sigma_depth)

    plt.imshow(object.resulting_image)
    plt.show()

    object.postprocess(blur = False,ds = True)

    plt.imshow(object.resulting_image)
    plt.show()

    filename = "test.png"
    skio.imsave(filename, np.uint8(object.resulting_image))