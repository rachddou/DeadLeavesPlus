import subprocess
import math
import logging
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from skimage.io import imread,imsave
from skimage.metrics import structural_similarity as ssim
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from time import time
# from scipy.fftpack import dct

def rgb_2_grey_tensor(tensor, train):
    """
    turn a rgb tensor in a grey scale image
    """
    #Y = 0.2125 R + 0.7154 G + 0.0721 B
    n,c,h,w = tensor.shape
    
    grey_scale = torch.zeros((n,1,h,w))
    if train:
        grey_scale = grey_scale.cuda()
    grey_scale[:,0,:,:] = 0.2126*tensor[:,0,...]+0.7152*tensor[:,1,...]+0.0722*tensor[:,2,...]
    return(grey_scale)

def var_map(input, w_size):
    """
    function that generates the local variance map of the input image
    w_size has to be odd
    """
    N, _, H, W = input.size()
    r = w_size//2
    input = torch.mean(input,dim=1).view(N,1,H,W)
    padded_img = nn.functional.pad(input,(r,r,r,r),mode = "replicate")
    unfold = torch.nn.Unfold(kernel_size=(w_size,w_size))
    output = unfold(padded_img)
    output = 2*torch.std(output,dim = 1).view((N,1,H,W))
    return(output)

def dct_map(img, normalisation_mode = "none"):
    """
    input : color image
    output : single map of dct criterion
    """
    N, _, H, W = img.size()
    img = torch.mean(img,dim=1).reshape(N,1,H,W)
    img = nn.functional.pad(img,(3,4,3,4),mode = "replicate")
    unfold = torch.nn.Unfold(kernel_size=(8,8))
    img = unfold(img)
    img = img.transpose(2,1)
    #mean_1 = img.mean(dim = -1,keepdim=True).expand(-1,-1,64)
    #img-= mean_1
    
    F = dct(np.eye(8), type=2, norm='ortho')
    FF = torch.Tensor(np.kron(F,F)).cuda()
    img = torch.abs(torch.matmul(img,FF))

    if normalisation_mode == "first exclusion":
        img = img[:,:,1:]
        norm = img.sum(-1, keepdim=True)
        img = img/(norm+1e-5)
    elif normalisation_mode == "classic":
        norm = img.sum(-1, keepdim=True)
        img = img/(norm+1e-5)
    elif normalisation_mode == "first = 1":
        norm = img[:,:,0].view(1,H*W,1).repeat(1,1,64)
        img = img/(norm+1e-5)
    img = torch.sort(img,dim = -1).values
    img = img[:,:,:32].sum(dim = -1).view((N,1,H,W))
    if normalisation_mode not in ["first exclusion","classic"]:
        pass
    img = (img -img.min())/(img.max()-img.min())
    return(img)

def compute_noise_map(input, nsigma, sca = 2,mode = "constant"):
    N, C, H, W = input.size()
    Hout = H//sca
    Wout = W//sca
    test = False
    if mode == "constant":
        noise_map = nsigma.view(N, 1, 1, 1).repeat(1, C, Hout, Wout)
        return(noise_map)
    elif mode == "single":
        noise_map = nsigma.view(N, 1, 1, 1).repeat(1, 1, Hout, Wout)
        return(noise_map)
    elif mode == "texture":
        nmin,nmax = 0.85*nsigma, 1.0*nsigma
        v_map = var_map(input,5)
        dct_img = dct_map(input, normalisation_mode = "classic")
        noise_map = dct_img*(1-v_map)

        #applying median filtering
        filtering = True
        if filtering :
            _, _, H, W = noise_map.size()
            noise_map = nn.functional.pad(noise_map,(2,2,2,2),mode = "replicate")
            unfolded = torch.nn.Unfold(kernel_size=(5,5))(noise_map)
            unfolded = unfolded.transpose(2,1)
            noise_map = torch.median(unfolded,dim = -1,keepdim=True)[0].view((1,1,H,W))

        omin, omax = noise_map.min(),noise_map.max()
        noise_map = nmin+ (nmax-nmin)*(1-(noise_map-omin)/(omax-omin))
        #noise_map = (1+ (noise_map-noise_map.mean())/(22*noise_map.std()))*nsigma
        noise_map = noise_map[:,:,0::2,0::2]
        noise_map = noise_map.repeat(1,C,1,1)
        if test:
            cpu_dct = dct_img.cpu().detach().numpy()[0,0]
            cpu_vmap = v_map.cpu().detach().numpy()[0,0]
            imsave("dct_map.png",cpu_dct)
            imsave("var_map.png",1-cpu_vmap)
        return(noise_map)








		

def weights_init_kaiming(lyr):
    r"""Initializes weights of the model according to the "He" initialization
    method described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015), using a
    normal distribution.
    This function is to be called by the torch.nn.Module.apply() method,
    which applies weights_init_kaiming() to every layer of the model.
    """
    classname = lyr.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(lyr.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(lyr.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        lyr.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).\
            clamp_(-0.025, 0.025)
        nn.init.constant(lyr.bias.data, 0.0)

def batch_psnr(img, imclean, data_range):
    r"""
    Computes the PSNR along the batch dimension (not pixel-wise)

    Args:
        img: a `torch.Tensor` containing the restored image
        imclean: a `torch.Tensor` containing the reference image
        data_range: The data range of the input image (distance between
            minimum and maximum possible values). By default, this is estimated
            from the image data-type.
    """
    img_cpu = img.data.cpu().numpy().astype(np.float32)
    imgclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(img_cpu.shape[0]):
        psnr += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :], \
                       data_range=data_range)
    return psnr/img_cpu.shape[0]

def batch_ssim(img, imclean):
    r"""
    Computes the PSNR along the batch dimension (not pixel-wise)

    Args:
        img: a `torch.Tensor` containing the restored image
        imclean: a `torch.Tensor` containing the reference image
        data_range: The data range of the input image (distance between
            minimum and maximum possible values). By default, this is estimated
            from the image data-type.
    """
    img_cpu = img.data.cpu().numpy().astype(np.float32)
    imgclean = imclean.data.cpu().numpy().astype(np.float32)
    ssim_val = 0
    for i in range(img_cpu.shape[0]):
        #print(imgclean[i, :, :, :].shape)
        ssim_val += ssim(np.transpose(imgclean[i, :, :, :],(1,2,0)), np.transpose(img_cpu[i, :, :, :], (1,2,0)),multichannel=True,channel_axis = 2,data_range=img_cpu[i,:,:,:].max() - img_cpu[i,:,:,:].min() )
    return ssim_val/img_cpu.shape[0]
def data_augmentation(image, mode):
    r"""Performs dat augmentation of the input image

    Args:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
            0 - no transformation
            1 - flip up and down
            2 - rotate counterwise 90 degree
            3 - rotate 90 degree and flip up and down
            4 - rotate 180 degree
            5 - rotate 180 degree and flip
            6 - rotate 270 degree
            7 - rotate 270 degree and flip
    """
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')
    return np.transpose(out, (2, 0, 1))

def variable_to_cv2_image(varim):
    r"""Converts a torch.autograd.Variable to an OpenCV image

    Args:
        varim: a torch.autograd.Variable
    """
    nchannels = varim.size()[1]
    if nchannels == 1:
        res = (varim.data.cpu().numpy()[0, 0, :]*255.).clip(0, 255).astype(np.uint8)
    elif nchannels == 3:
        res = varim.data.cpu().numpy()[0]
        res = cv2.cvtColor(res.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        #res = res.transpose(1, 2, 0)
        res = (res*255.).clip(0, 255).astype(np.uint8)
    else:
        raise Exception('Number of color channels not supported')
    return res

def get_git_revision_short_hash():
    r"""Returns the current Git commit.
    """
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()

def init_logger(argdict):
    r"""Initializes a logging.Logger to save all the running parameters to a
    log file

    Args:
        argdict: dictionary of parameters to be logged
    """
    from os.path import join

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    fh = logging.FileHandler(join("TRAINING_LOGS/"+argdict.log_dir, 'log.txt'), mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    try:
        logger.info("Commit: {}".format(get_git_revision_short_hash()))
    except Exception as e:
        logger.error("Couldn't get commit number: {}".format(e))
    logger.info("Arguments: ")
    for k in argdict.__dict__:
        logger.info("\t{}: {}".format(k, argdict.__dict__[k]))

    return logger

def init_logger_ipol():
    r"""Initializes a logging.Logger in order to log the results after
    testing a model

    Args:
        result_dir: path to the folder with the denoising results
    """
    logger = logging.getLogger('testlog')
    logger.setLevel(level=logging.INFO)
    fh = logging.FileHandler('out.txt', mode='w')
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def init_logger_test(result_dir):
    r"""Initializes a logging.Logger in order to log the results after testing
    a model

    Args:
        result_dir: path to the folder with the denoising results
    """
    from os.path import join

    logger = logging.getLogger('testlog')
    logger.setLevel(level=logging.INFO)
    fh = logging.FileHandler(join(result_dir, 'log.txt'), mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def normalize(data):
    r"""Normalizes a unit8 image to a float32 image in the range [0, 1]

    Args:
        data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
    """
    return np.float32(data/255.)

def svd_orthogonalization(lyr):
    r"""Applies regularization to the training by performing the
    orthogonalization technique described in the paper "FFDNet:	Toward a fast
    and flexible solution for CNN based image denoising." Zhang et al. (2017).
    For each Conv layer in the model, the method replaces the matrix whose columns
    are the filters of the layer by new filters which are orthogonal to each other.
    This is achieved by setting the singular values of a SVD decomposition to 1.

    This function is to be called by the torch.nn.Module.apply() method,
    which applies svd_orthogonalization() to every layer of the model.
    """
    classname = lyr.__class__.__name__
    if classname.find('Conv') != -1:
        weights = lyr.weight.data.clone()
        c_out, c_in, f1, f2 = weights.size()
        dtype = lyr.weight.data.type()

        # Reshape filters to columns
        # From (c_out, c_in, f1, f2)  to (f1*f2*c_in, c_out)
        weights = weights.permute(2, 3, 1, 0).contiguous().view(f1*f2*c_in, c_out)

        # Convert filter matrix to numpy array
        weights = weights.cpu().numpy()
        if np.mean(np.isnan(weights)) !=0 :
            print("Nan weights")

        # SVD decomposition and orthogonalization
        mat_u, _, mat_vh = np.linalg.svd(weights, full_matrices=False)
        weights = np.dot(mat_u, mat_vh)

        # As full_matrices=False we don't need to set s[:] = 1 and do mat_u*s
        lyr.weight.data = torch.Tensor(weights).view(f1, f2, c_in, c_out).\
            permute(3, 2, 0, 1).type(dtype)
    else:
        pass

def remove_dataparallel_wrapper(state_dict):
    r"""Converts a DataParallel model to a normal one by removing the "module."
    wrapper in the module dictionary

    Args:
        state_dict: a torch.nn.DataParallel state dictionary
    """
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, vl in state_dict.items():
        name = k[7:] # remove 'module.' of DataParallel
        new_state_dict[name] = vl

    return new_state_dict

def is_rgb(im_path):
    r""" Returns True if the image in im_path is an RGB image
    """
    from skimage.io import imread
    rgb = False
    im = imread(im_path)
    if (len(im.shape) == 3):
        if not(np.allclose(im[...,0], im[...,1]) and np.allclose(im[...,2], im[...,1])):
            rgb = True
    print("rgb: {}".format(rgb))
    print("im shape: {}".format(im.shape))
    return rgb

def contrast_reduction_(img,interval):
    """
    takes in a C*H*W image
    functions that apply a simple linear transform to reduce the contrast and dynamic of an image"""

    mu = img.mean(axis = (1,2))

    res = np.zeros(img.shape)
    res[0,...] = mu[0]+ interval*(img[0,...]-mu[0])
    res[1,...] = mu[1]+ interval*(img[1,...]-mu[1])
    res[2,...] = mu[2]+ interval*(img[2,...]-mu[2])
    return(res)
