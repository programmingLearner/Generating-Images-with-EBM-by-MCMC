# MCMC, 2020-202C Final, Tianyang Zhao
# Reference Paper: Learning Generative ConvNets via Multi-grid Modeling and Sampling

import os
import sys
import copy
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models


# For Colab
class Args:

	# preamble
	device = 0
	if_cudnn = True
	seed = 1
	deterministic = True

	# path
	dataset_dir = './'
	outputs_dir = './'
	reload_dir = './'

	# training
	batch_size_train = 100
	batch_size_test = 100	# please set same to train
	reload = True
	n_epochs = 100
	pre_train_epochs = 0
	log_interval = 50

	# MCMC, momentum?
	# weight decay?  clip?
	std_init_y_0 = 1e-1	# std for y_0, N(0, std)
	delta_tau = 3e-1	# Langevin learning rate
	delta_tau_decay = 0.995
	sigma = 28.0	# std for reference N(0, std), \sqrt{784} * 1.0
	std_z_tau = 1.8e-1	# std for Langevin z_tau, try tune down?
	MC_steps = 80	# 150
	MC_neg_step = 50	# effective sampling from this MC step, 120
	vis_step = 26
	MC_lr = 2.5e-3	# 5.0, start from 3e-2, for 12 + 12 epochs
	MC_lr_decay = 0.92	# decay per epoch, 0.7
	MC_weight_decay = 1e-4
	grad_clip = 1.0
	init_from_data_ratio = 0.15
	init_from_data_decay = 0.92
	
	# optim
	optim = 'SGD'
	optim_lr = 5e-2
	optim_betas = (0.9, 0.999)
	optim_eps = 1e-8
	optim_weight_decay = 3e-5
	optim_momentum = 0.0

	# network
	dim_list = [784,400,200,100,50,10]
	W_init = 1e-2	# 0 will raise problem of trivial convergence in multiple layers??
	lambda_k = 1e-5
	alpha = 3e-2


class Nets(nn.Module):
	def __init__(self):
		super(Nets, self).__init__()

		# Conv + FC Baseline, without offset
		self.Conv_W1 = nn.Parameter(torch.rand(5, 1, 5, 5) * 2e-2, requires_grad=True) 
		self.Conv_W2 = nn.Parameter(torch.rand(10, 5, 3, 3) * 2e-2, requires_grad=True)
		self.Conv_W3 = nn.Parameter(torch.rand(10, 10, 3, 3) * 2e-2, requires_grad=True)
		self.Linear_W = nn.Parameter(torch.rand(100, 10*3*3) * 2e-2, requires_grad=True)
		self.Classfication_W = nn.Parameter(torch.rand(10, 100) * 2e-2, requires_grad=True)
		self.Energy_W = nn.Parameter(torch.rand(1, 100) * 2e-2, requires_grad=True)

		# MC
		self.L_pos_grad = []
		self.L_neg_grad = []
		self.E_pos_grad = []
		self.E_neg_grad = []

		# self.C1_pos_grad = []
		# self.C2_pos_grad = []
		# self.C3_pos_grad = []
		# self.C1_neg_grad = []
		# self.C2_neg_grad = []
		# self.C3_neg_grad = []

	def forward(self, inputs, if_cudnn, output_type, requires_grad=False):
			# ouput = 1: classfication; = 2: output neg energy f_theta
			# requires_grad if need grad w.r.t. y 
		self.inputs = inputs
		if self.training:	# important
			self.inputs.requires_grad = requires_grad 	# important
			if requires_grad == True:
				self.inputs.retain_grad()

		x = F.conv2d(self.inputs, self.Conv_W1, padding=0)
		x = F.max_pool2d(F.relu(x), 2)
		x = F.conv2d(x, self.Conv_W2, padding=1)
		x = F.max_pool2d(F.relu(x), 2)
		x = F.conv2d(x, self.Conv_W3, padding=1)
		x = F.max_pool2d(F.relu(x), 2)
		x = x.view(x.shape[0], -1, 1)
		x = F.relu(torch.matmul(self.Linear_W, x))

		if output_type == 1:
				# classification
			x = torch.matmul(self.Classfication_W, x)
			x = x.view(x.shape[0], -1)
			x = F.log_softmax(x)
			return x

		else:
				# energy
			eng = torch.matmul(self.Energy_W, x)
			eng = eng.view(eng.shape[0], -1)

			if eng is None:
					# exploded
				print(self.Energy_W)
				sys.exit(0)

			return eng

	def save_pos_grad(self, fix_extractor=True):
		self.L_pos_grad.append(copy.deepcopy(self.Linear_W.grad.data))
		self.E_pos_grad.append(copy.deepcopy(self.Energy_W.grad.data))
		
		# self.C1_pos_grad.append(copy.deepcopy(self.Conv_W1.grad.data))
		# self.C2_pos_grad.append(copy.deepcopy(self.Conv_W2.grad.data))
		# self.C3_pos_grad.append(copy.deepcopy(self.Conv_W3.grad.data))

	def save_neg_grad(self, fix_extractor=True):
		self.L_neg_grad.append(copy.deepcopy(self.Linear_W.grad.data))
		self.E_neg_grad.append(copy.deepcopy(self.Energy_W.grad.data))

		# self.C1_neg_grad.append(copy.deepcopy(self.Conv_W1.grad.data))
		# self.C2_neg_grad.append(copy.deepcopy(self.Conv_W2.grad.data))
		# self.C3_neg_grad.append(copy.deepcopy(self.Conv_W3.grad.data))

	def get_partial_y(self):
		return self.inputs.grad

	def update_theta(self, learning_rate, weight_decay, grad_clip):

		length = len(self.E_pos_grad)
		for grad in self.E_pos_grad:
			self.Energy_W.data += learning_rate / length * torch.clamp(grad, -grad_clip, grad_clip) 	# / batch_size?
		length = len(self.E_neg_grad)
		for grad in self.E_neg_grad:
			self.Energy_W.data -= learning_rate / length * torch.clamp(grad, -grad_clip, grad_clip)
		self.Energy_W.data -= learning_rate * weight_decay * self.Energy_W.data

		length = len(self.L_pos_grad)
		for grad in self.L_pos_grad:
			self.Linear_W.data += learning_rate / length * torch.clamp(grad, -grad_clip, grad_clip)		# / batch_size?
		length = len(self.L_neg_grad)
		for grad in self.L_neg_grad:
			self.Linear_W.data -= learning_rate / length * torch.clamp(grad, -grad_clip, grad_clip)
		self.Linear_W.data -= learning_rate * weight_decay * self.Linear_W.data

		# clear pos and neg grad list
		self.L_pos_grad = []
		self.L_neg_grad = []
		self.E_pos_grad = []
		self.E_neg_grad = []

		# conv_rate = 3e-2

		# length = len(self.C1_pos_grad)
		# for grad in self.C1_pos_grad:
		# 	self.Conv_W1.data += conv_rate * learning_rate / length * torch.clamp(grad, -grad_clip, grad_clip)		# / batch_size?
		# length = len(self.C1_neg_grad)
		# for grad in self.C1_neg_grad:
		# 	self.Conv_W1.data -= conv_rate * learning_rate / length * torch.clamp(grad, -grad_clip, grad_clip)
		# self.Conv_W1.data -= conv_rate * learning_rate * weight_decay * self.Conv_W1.data

		# length = len(self.C2_pos_grad)
		# for grad in self.C2_pos_grad:
		# 	self.Conv_W2.data += conv_rate * learning_rate / length * torch.clamp(grad, -grad_clip, grad_clip)		# / batch_size?
		# length = len(self.C2_neg_grad)
		# for grad in self.C2_neg_grad:
		# 	self.Conv_W2.data -= conv_rate * learning_rate / length * torch.clamp(grad, -grad_clip, grad_clip)
		# self.Conv_W2.data -= conv_rate * learning_rate * weight_decay * self.Conv_W2.data

		# length = len(self.C3_pos_grad)
		# for grad in self.C3_pos_grad:
		# 	self.Conv_W3.data += conv_rate * learning_rate / length * torch.clamp(grad, -grad_clip, grad_clip)		# / batch_size?
		# length = len(self.C3_neg_grad)
		# for grad in self.C3_neg_grad:
		# 	self.Conv_W3.data -= conv_rate * learning_rate / length * torch.clamp(grad, -grad_clip, grad_clip)
		# self.Conv_W3.data -= conv_rate * learning_rate * weight_decay * self.Conv_W3.data

		# self.C1_pos_grad = []
		# self.C2_pos_grad = []
		# self.C3_pos_grad = []
		# self.C1_neg_grad = []
		# self.C2_neg_grad = []
		# self.C3_neg_grad = []


def main(args):
	
	torch.backends.cudnn.enabled = args.if_cudnn
	if args.deterministic:
		torch.manual_seed(args.seed)
		torch.backends.cudnn.deterministic = True

	# dataset
	train_loader = torch.utils.data.DataLoader(
						torchvision.datasets.MNIST(args.dataset_dir, train=True, download=True,
							transform=torchvision.transforms.Compose([
							torchvision.transforms.ToTensor(),
							torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
							batch_size=args.batch_size_train, shuffle=True)

	test_loader = torch.utils.data.DataLoader(
						torchvision.datasets.MNIST(args.dataset_dir, train=False, download=True,
							transform=torchvision.transforms.Compose([
							torchvision.transforms.ToTensor(),
							torchvision.transforms.Normalize((0.1307,), (0.3081,))])), 
							batch_size=args.batch_size_test, shuffle=True)

	# model
	network = Nets()

	# Optimizer
	if args.optim == 'SGD':
		optimizer = optim.SGD(network.parameters(), lr=args.optim_lr, \
			momentum=args.optim_momentum, weight_decay=args.optim_weight_decay)
	elif args.optim == 'Adam':
		optimizer = optim.Adam(network.parameters(), lr=args.optim_lr, \
			betas=args.optim_betas, eps=args.optim_eps, weight_decay=args.optim_weight_decay)
	
	# for GPU usage
	if args.if_cudnn:
		def use_gpu():
			return torch.cuda.is_available()
		if use_gpu():
			network.cuda(args.device)

	# Reload
	if args.reload == True:
		map_location = 'cpu'
		# if args.if_cudnn:
		# 	map_location = 'gpu'
		network_state_dict = torch.load('{}model.pth'.format(args.reload_dir), map_location=map_location)
		network.load_state_dict(network_state_dict)
		# optimizer_state_dict = torch.load('{}optimizer.pth'.format(args.reload_dir), map_location=map_location)
		# optimizer.load_state_dict(optimizer_state_dict)

	# Placeholders
	train_losses = []
	train_counter = []
	test_losses = []
	test_counter = [i*len(train_loader.dataset) for i in range(args.n_epochs + 1)]

	# Train an epoch
	def train(epoch, classification):
		network.train()
		for batch_idx, (data, target) in enumerate(train_loader):

			if args.if_cudnn:
				data = data.type(torch.cuda.FloatTensor)
				target = target.type(torch.cuda.LongTensor)

			if classification == 1:
				optimizer.zero_grad()
				output = network(data, args.if_cudnn, 1)
				loss = F.nll_loss(output, target)
				loss.backward()
				optimizer.step()

			else:


				### MCMC
				y_list = []

				## Positive Samples
				optimizer.zero_grad()
				f_theta = torch.sum(network(data, args.if_cudnn, 2))
				f_theta.backward()
				network.save_pos_grad()	# TODO
				pos_f = f_theta

				## Negative Samples
				# Init y_0 from Gaussian, (b,c,h,w)
				y = torch.randn(args.batch_size_train, 1, 28, 28) * args.std_init_y_0
				if args.if_cudnn:
					y = y.type(torch.cuda.FloatTensor)

				# What about Init y from Real Data?
				if np.random.rand(1) < args.init_from_data_ratio:
					y += copy.deepcopy(data.data)

				delta_tau = args.delta_tau

				# Langevin Dynamics, Parallel Operating on Batch
				for iter_MC in range(args.MC_steps):
					delta_tau *= args.delta_tau_decay

					if iter_MC % args.vis_step == 0:
						y_list.append(np.array(y[0,0,:,:].detach().cpu()))

					# Calculate \partial / \partial y
					optimizer.zero_grad()	# does this work? deep copy?
					f_theta = torch.sum(network(y, args.if_cudnn, 2, requires_grad=True))	# not data!!!
					f_theta.backward()
					partial_y = copy.deepcopy(network.get_partial_y().detach().data)	# TODO
					neg_f = f_theta
					
					# Largevin Update y
					z_tau = torch.randn(args.batch_size_train, 1, 28, 28) * args.std_z_tau
					if args.if_cudnn:
						z_tau = z_tau.type(torch.cuda.FloatTensor)
					tmp = copy.deepcopy(y.detach().data) / (args.sigma ** 2) - partial_y
					y =  copy.deepcopy(y.detach().data) + np.sqrt(delta_tau) * z_tau - delta_tau / 2 * tmp

					# Store for Negative Gradients
					if iter_MC > args.MC_neg_step:
						# For computational efficiency, this is actually calculating the y before update, no forward again
						network.save_neg_grad()	# TODO

				## Update Parameter theta
				network.update_theta(args.MC_lr, args.MC_weight_decay, args.grad_clip)	# TODO

			if batch_idx % args.log_interval == 0:
				if classification == 1:
					print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
						epoch, batch_idx * len(data), len(train_loader.dataset),
						100. * batch_idx / len(train_loader), loss.item()))
					train_losses.append(loss.item())
					train_counter.append(
						(batch_idx*args.batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
					torch.save(network.state_dict(), '{}model.pth'.format(args.outputs_dir))    # '.../outputs/inhibition-2005/ckpt/model.pth'
					torch.save(optimizer.state_dict(), '{}optimizer.pth'.format(args.outputs_dir))
				else:
					print('Train Epoch: {} [{}/{} ({:.0f}%)]\tPositive Samples: {}\tNegative Samples: {}'.format(
						epoch, batch_idx * len(data), len(train_loader.dataset),
						100. * batch_idx / len(train_loader), pos_f.item(), neg_f.item()))

					## Visualize Sampled Negative Examples, i.e. y (batch)
					# TODO
					plt.subplot(1,4,1)
					plt.imshow(y_list[0], cmap='gray')
					plt.subplot(1,4,2)
					plt.imshow(y_list[1], cmap='gray')
					plt.subplot(1,4,3)
					plt.imshow(y_list[2], cmap='gray')
					plt.subplot(1,4,4)
					plt.imshow(y_list[3], cmap='gray')
					plt.show()
					torch.save(network.state_dict(), '{}model.pth'.format(args.outputs_dir))    # '.../outputs/inhibition-2005/ckpt/model.pth'
					torch.save(optimizer.state_dict(), '{}optimizer.pth'.format(args.outputs_dir))

	# Test an epoch
	def test():
		network.eval()
		test_loss = 0
		correct = 0
		with torch.no_grad():
			for data, target in test_loader:

				if args.if_cudnn:
					data = data.type(torch.cuda.FloatTensor)
					target = target.type(torch.cuda.LongTensor)

				output = network(data, args.if_cudnn, 1)
				test_loss += F.nll_loss(output, target, size_average=False).item()
				pred = output.data.max(1, keepdim=True)[1]
				correct += pred.eq(target.data.view_as(pred)).sum()
		test_loss /= len(test_loader.dataset)
		test_losses.append(test_loss)
		print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			test_loss, correct, len(test_loader.dataset),
			100. * correct / len(test_loader.dataset)))

	# Workflow
	test()
	for epoch in range(1, args.n_epochs + 1):
		if epoch <= args.pre_train_epochs:
			train(epoch, 1)
		else:
			train(epoch, 0)
			args.MC_lr *= args.MC_lr_decay
			args.init_from_data_ratio *= args.init_from_data_decay
		test()

	# Training Curve
	fig = plt.figure()
	plt.plot(train_counter, train_losses, color='blue')
	plt.scatter(test_counter, test_losses, color='red')
	plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
	plt.xlabel('number of training examples seen')
	plt.ylabel('negative log likelihood loss')
	plt.show()


args = Args()
main(args)

