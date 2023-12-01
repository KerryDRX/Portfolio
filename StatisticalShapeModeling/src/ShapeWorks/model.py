import torch
from torch import nn
import torch.nn.functional as F
import sys
import json
import numpy as np
from collections import OrderedDict
from DeepSSMUtils import net_utils

# model out: base_filters*16
# All embed=234/119/75/32/12/6 (99/97/95/90/80/70)
hidden1, hidden2 = None, None
base_filters = 12

def set_hidden_layer_sizes(num_latent):
	global hidden1, hidden2
	# hidden1, hidden2 = 384, 96
	if num_latent <= 12:
		hidden1, hidden2 = 96, 12
	elif num_latent <= 48:
		hidden1, hidden2 = 96, 48
	elif num_latent <= 96:
		hidden1, hidden2 = 144, 96
	elif num_latent <= 144:
		hidden1, hidden2 = 170, 144
	elif num_latent <= 192:
		hidden1, hidden2 = 192, 192
	else:
		diff = num_latent - 192
		hidden1, hidden2 = 192 + diff//3, 192 + 2*diff//3

class ConvolutionalBackbone(nn.Module):
	def __init__(self, img_dims):
		super(ConvolutionalBackbone, self).__init__()
		self.img_dims = img_dims
		# basically using the number of dims and the number of poolings to be used 
		# figure out the size of the last fc layer so that this network is general to 
		# any images
		self.out_fc_dim = np.copy(img_dims)
		padvals = [4, 8, 8]
		for i in range(3):
			self.out_fc_dim[0] = net_utils.poolOutDim(self.out_fc_dim[0] - padvals[i], 2)
			self.out_fc_dim[1] = net_utils.poolOutDim(self.out_fc_dim[1] - padvals[i], 2)
			self.out_fc_dim[2] = net_utils.poolOutDim(self.out_fc_dim[2] - padvals[i], 2)

		self.features = nn.Sequential(OrderedDict([
			('conv1', nn.Conv3d(1, base_filters, 5)),
			('bn1', nn.BatchNorm3d(base_filters)),
			('relu1', nn.PReLU()),
			('mp1', nn.MaxPool3d(2)),

			('conv2', nn.Conv3d(base_filters, base_filters*2, 5)),
			('bn2', nn.BatchNorm3d(base_filters*2)),
			('relu2', nn.PReLU()),
			('conv3', nn.Conv3d(base_filters*2, base_filters*4, 5)),
			('bn3', nn.BatchNorm3d(base_filters*4)),
			('relu3', nn.PReLU()),
			('mp2', nn.MaxPool3d(2)),

			('conv4', nn.Conv3d(base_filters*4, base_filters*8, 5)),
			('bn4', nn.BatchNorm3d(base_filters*8)),
			('relu4', nn.PReLU()),
			('conv5', nn.Conv3d(base_filters*8, base_filters*16, 5)),
			('bn5', nn.BatchNorm3d(base_filters*16)),
			('relu5', nn.PReLU()),
			('mp3', nn.MaxPool3d(2)),

			('flatten', net_utils.Flatten()),
			
			('fc1', nn.Linear(self.out_fc_dim[0]*self.out_fc_dim[1]*self.out_fc_dim[2]*base_filters*16, hidden1)),
			('relu6', nn.PReLU()),
			('fc2', nn.Linear(hidden1, hidden2)),
			('relu7', nn.PReLU()),
		]))

	def forward(self, x):
		x_features = self.features(x)
		return x_features

# Create the custom layer to output the 4 NIG parameters
class DenseNormalGamma(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dense = nn.Linear(in_features, out_features * 4)
        
    def evidence(self, x):
        return F.softplus(x)
    
    def forward(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = torch.split(output, self.out_features, -1)
        
        # mu, alpha, beta with a softplus, gamma with a linear
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        
        return mu, v, alpha, beta

class DeterministicEncoder(nn.Module):
	def __init__(self, num_latent, img_dims, loader_dir):
		super(DeterministicEncoder, self).__init__()
		if torch.cuda.is_available():
			device = 'cuda:0'
		else:
			device = 'cpu'
		self.device = device
		self.num_latent = num_latent
		self.img_dims = img_dims
		self.loader_dir = loader_dir
		self.ConvolutionalBackbone = ConvolutionalBackbone(self.img_dims)
		self.pca_pred = nn.Sequential(OrderedDict([
			('linear', DenseNormalGamma(hidden2, self.num_latent))
		]))

	def forward(self, x):
		x = self.ConvolutionalBackbone(x)
		dist_params = self.pca_pred(x)
		pca_load_unwhiten = net_utils.unwhiten_PCA_scores(dist_params[0], self.loader_dir, self.device)
		return dist_params, pca_load_unwhiten

class DeterministicLinearDecoder(nn.Module):
	def __init__(self, num_latent, num_corr):
		super(DeterministicLinearDecoder, self).__init__()
		self.num_latent = num_latent
		self.numL = num_corr
		self.fc_fine = nn.Linear(self.num_latent, self.numL*3)

	def forward(self, pca_load_unwhiten):
		corr_out = self.fc_fine(pca_load_unwhiten).reshape(-1, self.numL, 3)
		return corr_out
        
'''
Supervised DeepSSM Model
'''
class DeepSSMNet(nn.Module):
	def __init__(self, config_file):
		super(DeepSSMNet, self).__init__()
		if torch.cuda.is_available():
			device = 'cuda:0'
		else:
			device = 'cpu'
		self.device = device
		with open(config_file) as json_file: 
			parameters = json.load(json_file)
		self.num_latent = parameters['num_latent_dim']

		set_hidden_layer_sizes(self.num_latent)
		print(f'MLP layers: {base_filters*16} -> {hidden1} -> {hidden2} -> {self.num_latent}')

		self.loader_dir = parameters['paths']['loader_dir']
		loader = torch.load(self.loader_dir + "validation")
		self.num_corr = loader.dataset.mdl_target[0].shape[0]
		img_dims = loader.dataset.img[0].shape
		self.img_dims = img_dims[1:]
		# encoder
		if parameters['encoder']['deterministic']:
			self.encoder = DeterministicEncoder(self.num_latent, self.img_dims, self.loader_dir)
		if not self.encoder:
			print("Error: Encoder not implemented.")
		# decoder
		if parameters['decoder']['deterministic']:
			if parameters['decoder']['linear']:
				self.decoder = DeterministicLinearDecoder(self.num_latent, self.num_corr)
		if not self.decoder:
			print("Error: Decoder not implemented.")

	def forward(self, x):
		dist_params, pca_load_unwhiten = self.encoder(x)
		corr_out = self.decoder(pca_load_unwhiten)
		return [dist_params, corr_out]