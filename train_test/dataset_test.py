
"""
Denoise an image with the FFDNet denoising method

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import argparse
from time import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.models import FFDNet, UNetRes
from skimage.transform import pyramid_reduce,rotate

# from utils.frequential_loss_v2 import *
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.utils import batch_ssim, normalize, init_logger_ipol, \
				variable_to_cv2_image, remove_dataparallel_wrapper, is_rgb, compute_noise_map,rgb_2_grey_tensor

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import skimage.io as skio


def test_ffdnet_dataset(**args):
    torch.manual_seed(0)
    r"""Denoises an input image with FFDNet
    """
    # Init logger
    logger = init_logger_ipol()
    # Check if input exists and if it is RGB
    im_files = [os.path.join(args['input'], f) for f in os.listdir(args['input']) if os.path.isfile(os.path.join(args['input'], f))]
    try:
        rgb_den = is_rgb(im_files[0])
    except:
        raise Exception('Could not open the input image')
    N_multi = args['Nmulti']

    if rgb_den:
        in_ch = 3
        model_fn = args['model_path']   
    else:
        # from HxWxC to  CxHxW grayscale image (C=1)
        in_ch = 1
        model_fn = args['model_path']
    #initialize global metrics
    psnr_test_set = 0
    ssim_test_set = 0
    acutance_test_set = 0
    PieAPP_test_set = 0
    PieAPP_test_set_n = 0
    acutance_test_set = 0
    noise_test = True
    # Absolute path to model file
    model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
                model_fn)

    # Create model
    print('Loading model ...\n')
    if args["model"] == "FFDNet":
        net = FFDNet(num_input_channels=in_ch)
        sca = 2
    elif args["model"]== "DRUNet":
        sca = 1
        if args["blind"]:
            net = UNetRes(in_nc = in_ch, out_nc = in_ch)
        else:    
            net = UNetRes(in_nc = in_ch+1, out_nc = in_ch)
            print("not blind")
    # Load saved weights
    if args['cuda']:
        if 'ckpt' in model_fn:
            state_dict = torch.load(model_fn)['state_dict']
        else:
            state_dict = torch.load(model_fn)
        if args["model"] == "FFDNet":
            model = nn.DataParallel(net).cuda()
        elif args["model"] == "DRUNet":
            model = net.cuda()
    else:
        state_dict = torch.load(model_fn, map_location='cpu')
        # CPU mode: remove the DataParallel wrapper
        state_dict = remove_dataparallel_wrapper(state_dict)
        model = net
    model.load_state_dict(state_dict)

    # Sets the model in evaluation mode (e.g. it removes BN)
    model.eval()

    # Sets data type according to CPU or GPU modes
    if args['cuda']:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    for filename in im_files:
        # Open image as a CxHxW torch.Tensor

        imorig = cv2.imread(filename)
        imorig = normalize(cv2.cvtColor(imorig, cv2.COLOR_BGR2RGB))

        if args["scale_invariance"]:
            imorig = pyramid_reduce(imorig,4.5,channel_axis=2)
        imorig = imorig[:1000,:1000,:]

        # from HxWxC to CxHxW, RGB image
        h = 8*(imorig.shape[0]//8)
        w = 8*(imorig.shape[1]//8)
        
        if args["ud_invariance"]:
            imorig = np.flipud(imorig) 
        if args["lr_invariance"]:
            imorig = np.fliplr(imorig) 
        if args["rot_invariance"]:
#             imorig = rotate_CV(imorig,45)[70:-70,70:-70,:]
            alpha = args["angle"]
            imorig = rotate(imorig,alpha,resize = True, preserve_range = True, mode = "constant")
            mask_rotation = np.bool_(rotate(np.ones((imorig.shape)),alpha,resize = True,preserve_range = True, mode = "constant"))

        h = 8*(imorig.shape[0]//8)
        w = 8*(imorig.shape[1]//8)
        imorig = imorig.transpose(2, 0, 1)
        imorig = imorig[:,:h,:w]
        print(imorig.shape)
        if args["rot_invariance"]:
            mask_rotation = mask_rotation[:h,:w,:]

        imorig = np.expand_dims(imorig, 0)
        # Handle odd sizes
        expanded_h = False
        expanded_w = False
        sh_im = imorig.shape
        if sh_im[2]%2 == 1:
            expanded_h = True
            imorig = np.concatenate((imorig, \
                    imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

        if sh_im[3]%2 == 1:
            expanded_w = True
            imorig = np.concatenate((imorig, \
                    imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)

        if args['contrast_reduction']:
            ratios = np.zeros(3)
            means = np.zeros(3)
            for i in range(3):
                f = 0.2
                ratios[i] = f/(1e-8+np.abs(imorig[:,i,...].max() -imorig[:,i,...].min()))
                means[i]= imorig[:,i,...].mean()
                imorig[:,i,...] = f*(imorig[:,i,...] -imorig[:,i,...].min())/(1e-8+np.abs(imorig[:,i,...].max() -imorig[:,i,...].min()))
                imorig[:,i,...] = imorig[:,i,...]+0.5-imorig[:,i,...].mean()    
            imorig = np.clip(imorig,0,1)
        imorig = torch.Tensor(imorig.copy())
        # Add noise
        if args['add_noise']:

            noise = torch.FloatTensor(imorig.size()).\
                    normal_(mean=0, std=args['noise_sigma'])
            imnoisy = imorig + noise
        else:
            imnoisy = imorig.clone()

            # Test mode
        with torch.no_grad(): # PyTorch v0.4.0

            imorig, imnoisy = Variable(imorig.type(dtype)), \
                    Variable(imnoisy.type(dtype))
            nsigma = Variable(torch.FloatTensor([args['noise_sigma']]).type(dtype))

        # Measure runtime
        start_t = time()

        # Estimate noise and subtract it to the input image
        if not(args["blind"]):
            noise_map = compute_noise_map(imnoisy, nsigma,sca, mode = args["noise_map"])
            
            if args["model"] == "FFDNet":
                out = model(imnoisy, noise_map)
            else:
                input_tensor = torch.cat((imnoisy, noise_map), dim=1)
                out = model(input_tensor)
        else:
        # Evaluate model and optimize it
            out = model(imnoisy)
        # out = model(imnoisy, noise_map)
        if args["residual"]:
            outim = torch.clamp(imnoisy-out, 0., 1.)
        else:
            outim = torch.clamp(out, 0., 1.)
        stop_t = time()

        if expanded_h:
            imorig = imorig[:, :, :-1, :]
            outim = outim[:, :, :-1, :]

        if expanded_w:
            imorig = imorig[:, :, :, :-1]
            outim = outim[:, :, :, :-1]
        
        fname = filename.split("/")
        fname = fname[-1].split(".")[0]
        print(fname)



                
        outimg = np.float64(variable_to_cv2_image(outim))

        
        if args['contrast_reduction']:
            outimg = outim[0,...].detach().cpu().numpy().transpose(1,2,0)
            for i in range(3):
                outimg[...,i] = (1./f)*(outimg[...,i]-outimg[...,i].min())
            outimg = np.clip(255*outimg,0,255).astype(np.uint8)
        img_orig = variable_to_cv2_image(imorig)
        residual = (img_orig - outimg)
        residual  = (residual -residual.min())/(residual.max() -residual.min())*255

        outimg = np.uint8(np.clip(outimg,0,255))
        psnr = compare_psnr(outimg, img_orig)
        image0, image1 = np.asarray(outimg,dtype = np.float32), np.asarray(img_orig,dtype = np.float32)
        if args["rot_invariance"]:
            mse = np.sum(mask_rotation*(image0-image1)**2)/np.sum(mask_rotation)
        else:
            mse = np.mean((image0-image1)**2)
        psnr = 10*np.log10((255**2)/mse)
            
        print(psnr)
        psnr_test_set+= psnr
            
        if args['save']:
            if not(os.path.isdir(os.path.join("tests",args["save_dir"]))):
                os.makedirs(os.path.join("tests",args["save_dir"]))

            cv2.imwrite("tests/"+args["save_dir"]+fname+"_ffdnet.png", outimg)
            if args['save_res']:
                cv2.imwrite("tests/"+args["save_dir"]+fname+"_residual.png", residual)

        # Compute PSNR and log it
        if rgb_den:
            logger.info("### RGB denoising ###")
        else:
            logger.info("### Grayscale denoising ###")
        if args['add_noise']:
            # psnr = batch_psnr(outim, imorig, 1.)
            # psnr_test_set+=psnr
            ssim_val = batch_ssim(outim,imorig)
            ssim_test_set+=ssim_val

        else:
            logger.info("\tNo noise was added, cannot compute PSNR")
        logger.info("\tRuntime {0:0.4f}s".format(stop_t-start_t))
    psnr_test_set*= (1./len(im_files))
    ssim_test_set*= (1./len(im_files))
    print("SSIM : {}".format(ssim_test_set))
    print("PSNR : {}".format(psnr_test_set))



