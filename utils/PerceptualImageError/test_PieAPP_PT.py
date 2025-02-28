import numpy as np
import cv2
import sys
import torch
from torch.autograd import Variable
sys.path.append('model/')
from model.PieAPPv0pt1_PT import PieAPP
sys.path.append('utils/')
from utils.image_utils import *
import argparse
import os
import skimage.io as skio
import matplotlib.pyplot as plt
from time import time
######## check for model and download if not present
if not os.path.isfile('weights/PieAPPv0.1.pth'):
	print ("downloading dataset")
	os.system("bash scripts/download_PieAPPv0.1_PT_weights.sh")
	if not os.path.isfile('weights/PieAPPv0.1.pth'):
		print ("PieAPPv0.1.pth not downloaded")
		sys.exit()

######## variables
patch_size = 64
batch_size = 1

######## input args
parser = argparse.ArgumentParser()
parser.add_argument("--ref_path", dest='ref_path', type=str, default='imgs/ref.png', help="specify input reference")
parser.add_argument("--A_path", dest='A_path', type=str, default='imgs/A.png', help="specify input image")
parser.add_argument("--sampling_mode", dest='sampling_mode', type=str, default='sparse', help="specify sparse or dense sampling of patches to compte PieAPP")
parser.add_argument("--save_map",dest = "save_map", action = "store_true", help = "specify if we save the weight and score maps")
parser.add_argument("--gpu_id", dest='gpu_id', type=str, default='0', help="specify which GPU to use", required=True)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

imagesA = np.expand_dims(cv2.imread(args.A_path),axis =0).astype('float32')
#print(imagesA.shape)
imagesRef = np.expand_dims(cv2.imread(args.ref_path),axis =0).astype('float32')
#print(imagesRef.shape)
_,rows,cols,ch = imagesRef.shape
print(imagesRef.shape)
if args.sampling_mode == 'sparse':
	stride_val = 27
else:
	stride_val = 1

try:
    gpu_num = float(args.gpu_id)
    use_gpu = 1
except ValueError:
    use_gpu = 0
except TypeError:
    use_gpu = 0
#print(cols)
y_loc = np.concatenate((np.arange(0, rows - patch_size, stride_val),np.array([rows - patch_size])), axis=0)
num_y = len(y_loc)
x_loc = np.concatenate((np.arange(0, cols - patch_size, stride_val),np.array([cols - patch_size])), axis=0)
num_x = len(x_loc)
print(num_x)
num_patches_per_dim = 10

######## initialize the model
PieAPP_net = PieAPP(int(batch_size),int(num_patches_per_dim))
#print(PieAPP_net)
PieAPP_net.load_state_dict(torch.load('weights/PieAPPv0.1.pth'))

if use_gpu == 1:
	PieAPP_net.cuda()

score_accum = 0.0
weight_accum = 0.0
t0 = time()
score_map = []
weight_map = []
# iterate through smaller size sub-images (to prevent memory overload)
for x_iter in range(0, -(-num_x//num_patches_per_dim)):
	score_map_x = []
	weight_map_x = []
	print(x_iter)
	for y_iter in range(0, -(-num_y//num_patches_per_dim)):
		# compute the size of the subimage
		if (num_patches_per_dim*(x_iter + 1) >= num_x):				
			size_slice_cols = cols - x_loc[num_patches_per_dim*x_iter]
		else:
			size_slice_cols = x_loc[num_patches_per_dim*(x_iter + 1)] - x_loc[num_patches_per_dim*x_iter] + patch_size - stride_val			
		if (num_patches_per_dim*(y_iter + 1) >= num_y):
			size_slice_rows = rows - y_loc[num_patches_per_dim*y_iter]
		else:
			size_slice_rows = y_loc[num_patches_per_dim*(y_iter + 1)] - y_loc[num_patches_per_dim*y_iter] + patch_size - stride_val
		# obtain the subimage and samples patches
		A_sub_im = imagesA[:, y_loc[num_patches_per_dim*y_iter]:y_loc[num_patches_per_dim*y_iter]+size_slice_rows, x_loc[num_patches_per_dim*x_iter]:x_loc[num_patches_per_dim*x_iter]+size_slice_cols,:]
		ref_sub_im = imagesRef[:, y_loc[num_patches_per_dim*y_iter]:y_loc[num_patches_per_dim*y_iter]+size_slice_rows, x_loc[num_patches_per_dim*x_iter]:x_loc[num_patches_per_dim*x_iter]+size_slice_cols,:]
		A_patches, ref_patches, temp_r, temp_c  = sample_patches(A_sub_im, ref_sub_im, patch_size=64, strideval=stride_val, random_selection=False, uniform_grid_mode = 'strided')
		num_patches_curr = A_patches.shape[0]/batch_size
		PieAPP_net.num_patches = num_patches_curr

		# initialize variable to be  fed to PieAPP_net
		A_patches_var = Variable(torch.from_numpy(np.transpose(A_patches,(0,3,1,2))), requires_grad=False)
		ref_patches_var = Variable(torch.from_numpy(np.transpose(ref_patches,(0,3,1,2))), requires_grad=False)
		if use_gpu == 1:
			A_patches_var = A_patches_var.cuda()
			ref_patches_var = ref_patches_var.cuda()

		# forward pass 
		_, PieAPP_patchwise_errors, PieAPP_patchwise_weights = PieAPP_net.compute_score(A_patches_var.float(), ref_patches_var.float())
		curr_err = PieAPP_patchwise_errors.cpu().data.numpy()
		curr_weights = 	PieAPP_patchwise_weights.cpu().data.numpy()
		score = np.multiply(curr_err,curr_weights)
		score_accum += np.sum(score)
		weight_accum += np.sum(curr_weights)
		# computing the maps
		w,l = len(temp_r),len(temp_c)
		tmp_err = score.reshape((w,l))
		curr_weights = curr_weights.reshape((w,l))
		score_map_x.append(tmp_err)
		weight_map_x.append(curr_weights)
	score_map_x = np.concatenate(score_map_x, axis = 0)
	weight_map_x = np.concatenate(weight_map_x,axis = 0)
	score_map.append(score_map_x)
	weight_map.append(weight_map_x)



if args.save_map:
    weight_map = np.concatenate(weight_map,axis = 1)
    score_map = np.concatenate(score_map,axis = 1)
    name = args.A_path.split("/")[1].split(".")[0]
    np.save("npfiles/"+name+"_weight_map.npy", weight_map)
    np.save("npfiles/"+name+"_score_map.npy", score_map)
    print(score_map.max())
    print(score_map.min())
    print(weight_map.max())
    print(weight_map.min())
    weight_map = (weight_map-weight_map.min())/(weight_map.max()-weight_map.min())
    score_map = (score_map-score_map.min())/(score_map.max()-score_map.min())
    name = args.A_path.split(".")[0]
    skio.imsave(name+"_weight_map.png",weight_map)
    skio.imsave(name+"_score_map.png",score_map)
print(time()-t0)
print ('PieAPP value of '+args.A_path+ ' with respect to: '+str(score_accum/weight_accum))
