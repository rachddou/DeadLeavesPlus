import torch.nn as nn
import torch
from torchvision.models import vgg19
import torchvision 

'''
EnhanceNet Implementation in PyTorch by Erik Quintanilla 

Single Image Super Resolution 

https://arxiv.org/abs/1612.07919/

This program assumes GPU.
'''

class Vgg_Features(nn.Module):
	def __init__(self, pool_layer_num = 9):
		'''
		To capture bothlow-level and high-level features, 
		we use a combination ofthe second and fifth pooling 
		layers and compute the MSEon their feature activations. 
		
		- Sajjadi et al.
		'''
		
		#we have maxpooling layers at [4,9,18,27,36]
		super(Vgg_Features, self).__init__()
		model = vgg19(pretrained=True)
		self.features = nn.Sequential(*list(model.features.children())[:pool_layer_num])

	def forward(self, img):
		return self.features(img)
		

		
		