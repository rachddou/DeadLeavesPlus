import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.io import read_image, write_png
import matplotlib.pyplot as plt
import random
from time import time
from omegaconf import OmegaConf
from dead_leaves_generation_pytorch.polygons_maker_bis import binary_polygon_generator, make_rectangle_mask
from dead_leaves_generation_pytorch.utils.texture_generation import bilevelTextureMixer, pattern_patch_two_colors
from dead_leaves_generation_pytorch.utils.utils import theoretical_number_disks, linear_color_gradient
from dead_leaves_generation_pytorch.utils.colored_noise import sample_color_noise
from dead_leaves_generation_pytorch.utils.perspective import perspective_shift
from dead_leaves_generation_pytorch.utils.blur import gaussian_blur_torch, lens_blur_torch
from dead_leaves_generation_pytorch.utils.color_conversion import rgb2lab_torch, lab2rgb_torch

# Load precomputed disk masks
dict_path = 'npy/dict.pt'
if os.path.exists(dict_path):
    dict_instance = torch.load(dict_path)
else:
    dict_instance = {}

class Textures:
    def __init__(self, width=1000, natural=True, path="",
                 texture_types=["sin"], texture_type_frequency=[1],
                 slope_range=[0.5, 2.5],
                 img_source=None,
                 warp=True, rdm_phase=False, device=None):

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.width = width
        self.natural = natural
        self.path = path
        self.warp = warp
        self.rdm_phase = rdm_phase
        self.gen = True

        if img_source is None:
            img_source = torch.randint(0, 256, (1000, 1000, 3), dtype=torch.uint8, device=self.device)
        elif not isinstance(img_source, torch.Tensor):
            img_source = torch.from_numpy(img_source).to(self.device)
        
        self.img_source = img_source
        self.w = img_source.shape[0]
        self.l = img_source.shape[1]
        self.files = []
        self.resulting_image = torch.ones((width, width, 3), dtype=torch.uint8, device=self.device)
        self.perspective_shift_enabled = False
        self.perspective_var = True
        
        if self.natural:
            self.source_image_sampling()
        self.texture_type_lists = texture_types
        self.texture_type_frequency = texture_type_frequency
        self.slope_range = slope_range
        self.texture_type = texture_types[0]
        self.sample_texture_type()

    def _sample_slope_from_ranges(self):
        if isinstance(OmegaConf.to_object(self.slope_range[0]), (list, tuple)):
            intervals = self.slope_range
            lengths = [interval[1] - interval[0] for interval in intervals]
            total_length = sum(lengths)
            interval_probs = torch.tensor([length / total_length for length in lengths], device=self.device)
            chosen_interval = torch.multinomial(interval_probs, 1).item()
            interval = intervals[chosen_interval]
            slope = torch.rand(1, device=self.device).item() * (interval[1] - interval[0]) + interval[0]
            return slope
        else:
            return torch.rand(1, device=self.device).item() * (self.slope_range[1] - self.slope_range[0]) + self.slope_range[0]

    def lin_gradient(self, color1, color2, angle=45):
        k = torch.rand(1, device=self.device).item() * 0.4 + 0.1
        textureMap = linear_color_gradient(color_1=color1, color_2=color2, width=self.width, angle=angle, k=k, color_space="lab", device=self.device)
        return textureMap
    
    def random_patch_selection(self):
        x = torch.randint(0, max(1, self.w - 101), (1,), device=self.device).item()
        y = torch.randint(0, max(1, self.l - 101), (1,), device=self.device).item()
        random_patch = self.img_source[x:x+100, y:y+100]
        return random_patch
    
    def generate_source_texture(self, width=1000):
        if self.texture_type == "freq_noise":
            colorNoiseSlope = self._sample_slope_from_ranges()
            textureMap = sample_color_noise(self.random_patch_selection(), width=width, slope=colorNoiseSlope, device=self.device)
        else:
            if self.texture_type == "texture_mixes":
                singleColor1 = torch.rand(1, device=self.device).item() < 0.1
                singleColor2 = torch.rand(1, device=self.device).item() < 0.1
                textureMixMode = [["sin"], ["grid"]]
                mixMode = torch.multinomial(torch.tensor([0.9, 0.1], device=self.device), 1).item()
                textureMixMode = textureMixMode[mixMode]
            else:
                singleColor1 = True
                singleColor2 = True
                textureMixMode = [self.texture_type]
            
            if self.warp:
                warp = torch.rand(1, device=self.device).item() < 0.5
            
            thresh = torch.randint(5, 50, (1,), device=self.device).item()
            textureMap = bilevelTextureMixer(
                color_source_1=self.random_patch_selection(),
                color_source_2=self.random_patch_selection(),
                single_color1=singleColor1, single_color2=singleColor2,
                mixing_types=textureMixMode, width=width,
                thresh_val=thresh, warp=warp,
                slope_range=self.slope_range, device=self.device)
        
        return textureMap
    
    def source_image_sampling(self):
        """Selects a random source image from the specified directory."""
        if not self.files:
            self.files = [os.path.join(self.path, f) for f in os.listdir(self.path) 
                         if os.path.isfile(os.path.join(self.path, f)) 
                         and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        N = len(self.files)
        ind = random.randint(0, N-1)
        try:
            # Read image directly to GPU if available
            img = read_image(self.files[ind])
            
            # Convert to device immediately after reading
            img = img.to(self.device)
            
            if img.shape[0] == 1:  # Grayscale
                img = img.repeat(3, 1, 1)
            elif img.shape[0] == 4:  # RGBA
                img = img[:3, :, :]  # Drop alpha channel
            
            self.img_source = img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
            self.w = self.img_source.shape[0]
            self.l = self.img_source.shape[1]
            print(self.files[ind])
        except Exception as e:
            print(f"Error loading image: {e}")
            self.source_image_sampling()
        
    def sample_texture_type(self):
        if len(self.texture_type_lists) == 1:
            self.texture_type = self.texture_type_lists[0]
        else:
            probs = torch.tensor(self.texture_type_frequency, device=self.device, dtype=torch.float32)
            idx = torch.multinomial(probs, 1).item()
            self.texture_type = self.texture_type_lists[idx]
            
    def generate_texture(self, width=100):
        t0 = time()
        self.sample_texture_type()
        tmp_rdm_phase = self.rdm_phase
        if not self.gen:
            print(self.texture_type)
        if self.texture_type == "gradient":
            x1, y1 = torch.randint(0, self.w, (1,), device=self.device).item(), torch.randint(0, self.l, (1,), device=self.device).item()
            x2, y2 = torch.randint(0, self.w, (1,), device=self.device).item(), torch.randint(0, self.l, (1,), device=self.device).item()
            color1 = self.img_source[x1, y1, :].byte()
            color2 = self.img_source[x2, y2, :].byte()
            res = self.lin_gradient(color1, color2, angle=45)
        else:
            if self.texture_type in ["grid", "texture_mixes"] and self.rdm_phase:
                self.rdm_phase = False
            single_color = torch.rand(1, device=self.device).item() > 0.5
            self.perspective_var = torch.rand(1, device=self.device).item() > 0.5
            if self.perspective_var and self.perspective_shift_enabled:
                res = self.generate_source_texture(width=2*width)
                print("perspective shifting")
                res = perspective_shift(res, device=self.device)
            else:
                res = self.generate_source_texture(width=width)
            self.rdm_phase = tmp_rdm_phase
        return res

    def clear(self):
        self.resulting_image = torch.ones((self.width, self.width, 3), dtype=torch.uint8, device=self.device)


class Deadleaves(Textures):
    def __init__(self, rmin=1, rmax=1000, alpha=3, width=1000, natural=True, path="",
                 texture_path="", shape_type="poly", texture_types=["sin"], texture_type_frequency=[1],
                 slope_range=[0.5, 2.5], texture=True, gen=False, warp=True, rdm_phase=False, perspective=True,
                 img_source=None, device=None):
        
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(width=width, natural=natural, path=path, texture_types=texture_types,
                        texture_type_frequency=texture_type_frequency, slope_range=slope_range,
                        img_source=img_source, warp=warp, rdm_phase=rdm_phase, device=self.device)
        
        self.rmin = rmin
        self.rmax = rmax
        self.alpha = alpha
        self.texture_path = texture_path
        self.shape_type = shape_type
        self.texture = texture
        self.perspective_shift_enabled = perspective
        self.gen = gen
        self.n_textures = 10
        self.textures = []

        self.binary_image = torch.ones((width, width), dtype=torch.bool, device=self.device)
        self.resulting_image = torch.ones((width, width, 3), dtype=torch.uint8, device=self.device)

        self.vamin = 1/(rmax**(alpha-1))
        self.vamax = 1/(rmin**(alpha-1))
        theoretical_n_shapes = theoretical_number_disks(0.002, rmin, rmax, width, alpha)
        interval = int(theoretical_n_shapes)
        print(interval)
        if interval < 10:
            interval = 1
        if interval > 10:
            interval = interval // 10
        self.interval = interval

    def update_leaves_size_parameters(self, rmin, rmax):
        self.rmin = rmin
        self.rmax = rmax
        self.vamin = 1/(self.rmax**(self.alpha-1))
        self.vamax = 1/(self.rmin**(self.alpha-1))
        theoretical_n_shapes = theoretical_number_disks(0.002, self.rmin, self.rmax, self.width, self.alpha)
        interval = int(theoretical_n_shapes)
        print(interval)
        if interval < 10:
            interval = 1
        if interval > 10:
            interval = interval // 10
        self.interval = interval

    def fetch_textures(self):
        files = [os.path.join(self.texture_path, f) for f in os.listdir(self.texture_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        file_id = torch.randint(0, len(files), (self.n_textures,), device='cpu').numpy()
        textures = []
        for ind in file_id:
            try:
                img = read_image(files[ind])
                # Convert to device immediately
                img = img.to(self.device)
                
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
                elif img.shape[0] == 4:  # RGBA
                    img = img[:3, :, :]
                
                textures.append(img.permute(1, 2, 0))
            except Exception as e:
                print(f"Error loading texture: {e}")
                continue
        self.textures = textures
        return textures

    def generate_textures_dictionary(self):
        textures = []
        for _ in range(self.n_textures):
            angle = torch.randint(0, 360, (1,), device=self.device).item()
            texture_img = self.generate_texture(width=2*self.rmax+100)
            scale_dict = {"1": texture_img}
            
            for i in range(2, 6):
                h, w = texture_img.shape[0], texture_img.shape[1]
                new_h, new_w = h // i, w // i
                if texture_img.dim() == 3:
                    texture_resized = F.interpolate(
                        texture_img.permute(2, 0, 1).unsqueeze(0).float(),
                        size=(new_h, new_w),
                        mode='area'
                    ).squeeze(0).permute(1, 2, 0).byte()
                else:
                    texture_resized = F.interpolate(
                        texture_img.unsqueeze(0).unsqueeze(0).float(),
                        size=(new_h, new_w),
                        mode='area'
                    ).squeeze().byte()
                scale_dict[str(i)] = texture_resized
            textures.append(scale_dict)
        self.textures = textures

    def pick_texture(self, size):
        idx = torch.randint(0, self.n_textures, (1,), device=self.device).item()
        current_texture_dict = self.textures[idx]

        h = current_texture_dict["1"].shape[0]
        max_scale = min(5, h/size)
        scale = torch.floor(1 + max(0, (max_scale-1)*torch.rand(1, device=self.device).item()**(2/3))).int().item()
        current_texture = current_texture_dict[str(scale)]

        transform = torch.randint(0, 4, (1,), device=self.device).item()
        if transform == 0:
            current_texture = torch.flip(current_texture, [0])
        elif transform == 1:
            current_texture = torch.flip(current_texture, [1])
        elif transform == 2:
            current_texture = torch.rot90(current_texture, k=1, dims=(0, 1))
        elif transform == 3:
            current_texture = torch.flip(torch.rot90(current_texture, k=1, dims=(0, 1)), [0])
        return current_texture

    def resize_textures(self, size, texture):
        h = texture.shape[0]
        max_scale = min(5, h/size)
        scale = 1 + (max_scale-1)*torch.rand(1, device=self.device).item()**(2/3)
        scale = int(scale*5)/5.
        
        if scale > 1:
            new_h, new_w = int(h / scale), int(texture.shape[1] / scale)
            if texture.dim() == 3:
                texture_resized = F.interpolate(
                    texture.permute(2, 0, 1).unsqueeze(0).float(),
                    size=(new_h, new_w),
                    mode='area'
                ).squeeze(0).permute(1, 2, 0).byte()
            else:
                texture_resized = F.interpolate(
                    texture.unsqueeze(0).unsqueeze(0).float(),
                    size=(new_h, new_w),
                    mode='area'
                ).squeeze().byte()
        else:
            texture_resized = texture
        
        return texture_resized

    def generate_single_shape_mask(self):
        radius = self.vamin + (self.vamax-self.vamin)*torch.rand(1, device=self.device).item()
        radius = int(1/(radius**(1./(self.alpha-1))))
        shape = self.shape_type
        if shape == "mix":
            shape = random.choice(["disk", "poly", "rectangle"])
        
        if shape == "poly":
            max_n = max(100, int(0.9*radius))
            n = torch.randint(50, max_n, (1,)).item() if max_n > 50 else 50
            shape_1d = binary_polygon_generator(
                2*(3*radius//2)+1, 
                n=n,
                allow_holes=bool(random.getrandbits(1)), 
                smoothing=bool(random.getrandbits(1)),
                device=self.device
            )
            
            if max(shape_1d.shape[0], shape_1d.shape[1]) >= 2*self.rmax+100:
                scale = (2*self.rmax+100.)/max(shape_1d.shape[0], shape_1d.shape[1])
                new_size = (int(2*((shape_1d.shape[0]*scale)//2)-1), int(2*((shape_1d.shape[1]*scale)//2)-1))
                shape_1d = F.interpolate(shape_1d.unsqueeze(0).unsqueeze(0).float(), size=new_size, mode='nearest').squeeze().bool()
        elif shape == "disk":
            if str(radius) in dict_instance:
                shape_1d = dict_instance[str(radius)].to(self.device)
            else:
                # Create disk mask on the fly
                y, x = torch.meshgrid(torch.arange(-radius, radius+1, device=self.device), 
                                     torch.arange(-radius, radius+1, device=self.device), indexing='ij')
                shape_1d = (x**2 + y**2) <= radius**2
        elif shape == "rectangle":
            shape_1d = make_rectangle_mask(radius, device=self.device)
            if max(shape_1d.shape[0], shape_1d.shape[1]) >= 2*self.rmax+100:
                scale = (2*self.rmax+100.)/max(shape_1d.shape[0], shape_1d.shape[1])
                new_size = (int(2*((shape_1d.shape[0]*scale)//2)-1), int(2*((shape_1d.shape[1]*scale)//2)-1))
                shape_1d = F.interpolate(shape_1d.unsqueeze(0).unsqueeze(0).float(), size=new_size, mode='nearest').squeeze().bool()
        
        return shape_1d, radius

    def add_shape_to_binary_mask(self, shape_1d):
        width_shape, length_shape = shape_1d.shape[0], shape_1d.shape[1]
        pos = [torch.randint(0, self.width, (1,), device=self.device).item(),
               torch.randint(0, self.width, (1,), device=self.device).item()]
        
        x_min = max(0, pos[0]-width_shape//2)
        x_max = min(self.width, 1+pos[0]+width_shape//2)
        y_min = max(0, pos[1]-length_shape//2)
        y_max = min(self.width, 1+pos[1]+length_shape//2)

        shape_mask_1d = self.binary_image[x_min:x_max, y_min:y_max].clone()
        shape_1d = shape_1d[max(0, width_shape//2-pos[0]):min(width_shape, self.width+width_shape//2-pos[0]),
                           max(0, length_shape//2-pos[1]):min(length_shape, self.width+length_shape//2-pos[1])]

        shape_mask_1d = shape_mask_1d & shape_1d
        self.binary_image[x_min:x_max, y_min:y_max] = self.binary_image[x_min:x_max, y_min:y_max] & (~shape_mask_1d)
        return x_min, x_max, y_min, y_max, shape_mask_1d

    def render_shape(self, shape_mask_1d, radius):
        t = time()
        width_shape, length_shape = shape_mask_1d.shape[0], shape_mask_1d.shape[1]

        shape_mask = shape_mask_1d.unsqueeze(-1).repeat(1, 1, 3).float()
        shape_render = shape_mask.clone()
        
        x, y = torch.randint(0, self.w, (1,), device=self.device).item(), torch.randint(0, self.l, (1,), device=self.device).item()
        color = self.img_source[x, y, :].byte()
        
        if radius < 20:
            shape_render = color.float() * shape_render
        else:
            angle = torch.randint(0, 360, (1,), device=self.device).item()
            x2, y2 = torch.randint(0, self.w, (1,), device=self.device).item(), torch.randint(0, self.l, (1,), device=self.device).item()
            color_2 = self.img_source[x2, y2, :].byte()
            grad_vs_texture = torch.rand(1, device=self.device).item()
            if not self.texture:
                grad_vs_texture = 1
            if grad_vs_texture > 0.95:
                k = torch.rand(1, device=self.device).item() * 0.4 + 0.1
                textureMap = linear_color_gradient(color, color_2, width=2*max(width_shape, length_shape)+1, angle=angle, k=k, color_space="lab", device=self.device)
            else:
                if self.gen:
                    textureMap = self.generate_texture(width=60+max(width_shape, length_shape))
                else:
                    textureMap = self.pick_texture(size=max(width_shape, length_shape))

            h, w = textureMap.shape[0], textureMap.shape[1]
            x_tex = torch.randint(0, max(1, h - width_shape), (1,), device=self.device).item()
            y_tex = torch.randint(0, max(1, w - length_shape), (1,), device=self.device).item()
            textureMap = textureMap[x_tex:x_tex+width_shape, y_tex:y_tex+length_shape]
            
            # Ensure textureMap has correct shape
            if textureMap.shape[0] != width_shape or textureMap.shape[1] != length_shape:
                # Pad or crop if necessary
                if textureMap.dim() == 3:
                    textureMap = F.interpolate(
                        textureMap.permute(2, 0, 1).unsqueeze(0).float(),
                        size=(width_shape, length_shape),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0).byte()
                else:
                    textureMap = F.interpolate(
                        textureMap.unsqueeze(0).unsqueeze(0).float(),
                        size=(width_shape, length_shape),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze().byte()
            
            shape_render = (shape_render * textureMap.float()).byte()
        
        return shape_mask, shape_render

    def generate_stack(self, disk_count):
        for i in range(disk_count):
            shape_1d, radius = self.generate_single_shape_mask()
            x_min, x_max, y_min, y_max, shape_mask_1d = self.add_shape_to_binary_mask(shape_1d)
            shape_mask, shape_render = self.render_shape(shape_mask_1d, radius)
            
            self.resulting_image[x_min:x_max, y_min:y_max, :] = (
                self.resulting_image[x_min:x_max, y_min:y_max, :].float() * (1-shape_mask)
            ).byte()
            self.resulting_image[x_min:x_max, y_min:y_max, :] += shape_render.byte()
        
        print("dead_leaves stack created")
        return self.resulting_image.clone(), (1-self.binary_image.unsqueeze(-1).repeat(1, 1, 3)).byte().clone()

    def clear(self):
        self.resulting_image = torch.ones((self.width, self.width, 3), dtype=torch.uint8, device=self.device)
        self.binary_image = torch.ones((self.width, self.width), dtype=torch.bool, device=self.device)

    def compose_dead_leaves_depth_of_field(self, blur_type, blur_val, fetch=False):
        if self.natural:
            self.source_image_sampling()
            if self.img_source.shape[2] != 3:
                self.source_image_sampling()
        if self.texture:
            if not self.gen:
                if fetch:
                    self.fetch_textures()
                else:
                    self.generate_textures_dictionary()

        background, back_mask = self.generate_stack(10*self.interval)
        self.clear()
        plainground, plain_mask = self.generate_stack(self.interval)
        self.clear()
        foreground, fore_mask = self.generate_stack(int(0.5*self.interval))

        if blur_type == "gaussian":
            background = gaussian_blur_torch(background, kernel_size=11, sigma=blur_val)
        elif blur_type == "lens":
            background = lens_blur_torch(background, radius=blur_val, components=2, exposure_gamma=2)
        
        plainground = ((1-plain_mask.float())*background.float() + plain_mask.float()*plainground.float()).byte()
        foreground = (foreground.float() + (1-fore_mask.float())*plainground.float()).byte()
        
        if blur_type == "gaussian":
            foreground = gaussian_blur_torch(foreground, kernel_size=11, sigma=blur_val)
            fore_mask = gaussian_blur_torch(fore_mask.float(), kernel_size=11, sigma=blur_val)
        elif blur_type == "lens":
            foreground = lens_blur_torch(foreground, radius=blur_val, components=2, exposure_gamma=2)
            fore_mask = lens_blur_torch(fore_mask.float()*255, radius=blur_val, components=2, exposure_gamma=2)/255.
        
        im2 = plainground.float() * (1-fore_mask.float())
        res = foreground.float() * fore_mask.float()
        im2 += res
        self.resulting_image = torch.clamp(im2, 0, 255).byte()

    def postprocess(self, blur=True, ds=True):
        if blur or ds:
            if blur:
                blur_value = torch.rand(1, device=self.device).item() * 2 + 1
                self.resulting_image = gaussian_blur_torch(self.resulting_image, kernel_size=11, sigma=blur_value)
            if ds:
                h, w = self.resulting_image.shape[0], self.resulting_image.shape[1]
                new_h, new_w = h // 2, w // 2
                self.resulting_image = F.interpolate(
                    self.resulting_image.permute(2, 0, 1).unsqueeze(0).float(),
                    size=(new_h, new_w),
                    mode='area'
                ).squeeze(0).permute(1, 2, 0).byte()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    object = Deadleaves(rmin=20, rmax=400, alpha=3,
                       width=1000, natural=True, path="/Users/raphael/Workspace/telecom/code/ffdnet_core/datasets/HIGH X2 Urban",
                       shape_type="mix", texture=True, device=device)

    yo = object.random_patch_selection()

    shapes = ["poly", "disk", "rectangle", "mix"]
    shape_type = "mix"
    object.shape_type = shape_type

    sigma_depth = 5*torch.rand(1, device=device).item()**(1/2)
    object.compose_dead_leaves_depth_of_field("gaussian", sigma_depth)

    # Convert to numpy for plotting
    img_np = object.resulting_image.cpu().numpy()
    plt.imshow(img_np)
    plt.show()

    object.postprocess(blur=False, ds=True)

    img_np = object.resulting_image.cpu().numpy()
    plt.imshow(img_np)
    plt.show()

    # Save using torchvision
    img_save = object.resulting_image.permute(2, 0, 1)
    write_png(img_save.cpu(), "test.png")