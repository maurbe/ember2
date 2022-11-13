import torch
import torch.nn as nn
import torch.nn.functional as F
import os, glob, tqdm, numpy as np

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sorcery import dict_of
from lib.nn.modules import *
from lib.nn.utils import *

import warnings
warnings.filterwarnings("ignore")


# ........................................................................


class net(nn.Module):

	def __init__(self, config, mode):
		super().__init__()

		self.mode = mode
		self.config = config
		cfg = self.get_cfg()

		self.G = Generator(cfg)
		self.D = Discriminator(**cfg)

		self.GE  = Generator(cfg)
		self.ema = EMA(0.995)

		self.aux = Generator(cfg) if mode=='zoom' else None
		self.configure_optimizers()
	

	def forward(self, x, c, noise=None):
		return self.GE(x, c, noise) # G or GE?


	def summary(self):
		print('Gen:', summary(self.G), flush=True)
		print('Dis:', summary(self.D), flush=True)


	def EMA(self):
		def update_moving_average(current_model, ma_model):
			for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
				old_weight, up_weight = ma_params.data, current_params.data
				ma_params.data = self.ema.update_average(old_weight, up_weight)
		update_moving_average(self.G, self.GE)


	#def decay(self, step):
	#	return 0.8 ** (step / 1e5)

	def configure_optimizers(self):

		lr = self.config['lr']
		self.opt_g = Adam(self.G.parameters(), lr=lr, betas=(0.0, 0.99))
		self.opt_d = Adam(self.D.parameters(), lr=lr, betas=(0.0, 0.99))

		#self.scheduler_g = StepLR(self.opt_g, step_size=1000001, gamma=0.1, verbose=True) #lr_lambda=self.decay, verbose=True)
		#self.scheduler_d = StepLR(self.opt_d, step_size=1000001, gamma=0.1, verbose=True) #lr_lambda=self.decay, verbose=True)


	def _step(self, x, y, c, level=0, zoom=False, do_opt=True):

		# train G
		self.opt_g.zero_grad(set_to_none=True)
		toggle_grad(self.G, True)
		toggle_grad(self.D, False)

		# here we need to generate 2 images with different noises!
		p1 = self.G(x, c)
		p2 = self.G(x, c)
	
		#r = self.D(pair(lowpass_resize(y) if zoom else x, y, p2, c))
		f = self.D(pair(x, p1, p2, c))
		
		g_loss = NSLoss(None, f, mode='G') # RAHingeLoss(r, f, mode='G')
		g_loss.backward()

		#Gradient Value Clipping due to high lr
		#nn.utils.clip_grad_value_(self.G.parameters(), clip_value=0.1)

		if do_opt:
			self.opt_g.step()
			#self.scheduler_g.step()

		self.log(f'g_adv_{level}', g_loss)
		if self.step % 1000 == 0:
			self.log_network(self.G, "G")

		
		# train D
		self.opt_d.zero_grad(set_to_none=True)
		toggle_grad(self.D, True)
		toggle_grad(self.G, False)

		p1 = p1.detach()
		p2 = p2.detach()
		
		# NOTE: the pairs to discriminate are: 
		# BASE: (x, p1, p2) and (x, y, p1)
		# ZOOM: (x, p1, p2) and (lowpass(y), y, p1)
		
		r = self.D(pair(lowpass_resize(y) if zoom else x, y, p2, c))
		f = self.D(pair(x, p1, p2, c))

		d_loss = NSLoss(r, f, mode='D') # RAHingeLoss(r, f, mode='D')
		d_loss.backward()

		#Gradient Value Clipping due to high lr
		#nn.utils.clip_grad_value_(self.D.parameters(), clip_value=0.1)
		
		if do_opt:
			self.opt_d.step()
			#self.scheduler_d.step()
		
		self.log(f'dfake_{level}', f)
		self.log(f'dtrue_{level}', r)
		self.log(f'd_adv_{level}', d_loss)
		if self.step % 1000 == 0:
			self.log_network(self.D, "D")
		
		return p1

	
	def train_step_base(self, batch):

		x = batch['x']
		y = batch['y']
		c = batch['c']
		p =  self._step(x, y, c) #, do_opt=(self.step%8==0))

		if self.step % 10 == 0 and self.step > 10000:
			self.EMA()


	def train_step_zoom(self, batch):

		x  = batch['x']
		ys = batch['y'].transpose(0, 1)
		cs = batch['c'].transpose(0, 1)
		qs = batch['q'].transpose(0, 1)

		# base prediction
		with torch.no_grad(): 
			# construct this context manually: keep redshift, assign res = 1.0
			c = torch.clone(cs[0])
			c[:, 1] = 1.0 # is this necessary or is it already 1.0??
			x = self.aux(x, c)

		levels = ys.shape[0]
		num_levels = np.searchsorted([3, 6, 7, 8, 9], self.step / 100000) + 1 

		for level, (y, c, q) in enumerate(zip(ys, cs, qs)):
			if level < num_levels:
				x = select_quadrant(x, q)
				x = self._step(x, y, c, level, zoom=True, 
							   do_opt=(level == num_levels - 1))

		if self.step % 10 == 0 and self.step > 10000:
			self.EMA()
		

	def train_step(self, step, batch, logger):
		self.step = step
		self.logger = logger

		if self.mode == 'base':
			self.train_step_base(batch)
		else:
			self.train_step_zoom(batch)
		self.logger = None


	def predict_base(self, data):
		self.eval()

		for snap in tqdm.tqdm(data.keys()):         
			x = data[snap]['x']
			c = data[snap]['c']
			data[snap]['p'] = self.GE(x, c, noise=None).detach() # G or GE?
		return data


	def discriminate_base(self, data):
		self.eval()

		for snap in tqdm.tqdm(data.keys()):         
			x = data[snap]['x']
			y = data[snap]['y']
			c = data[snap]['c']

			p1 = self.GE(x, c, noise=None).detach()
			p2 = self.GE(x, c, noise=None).detach()

			data[snap]['fakes'] = self.D(pair(x, p1, p2, c)).detach()
			data[snap]['reals'] = self.D(pair(x, y , p2, c)).detach()

			# free memory
			del data[snap]['x']
			del data[snap]['y']
			del data[snap]['c']
			
		return data


	def predict_zoom(self, data):
		self.eval()
		
		for snap in tqdm.tqdm(data.keys()):
			x  = data[snap]['x']
			cs = data[snap]['c']
			qs = data[snap]['q']

			c = torch.clone(cs[0])
			c[:, 1] = 1.0
			x = self.aux(x, c)
			data[snap]['b'] = x.detach()
			
			out = []
			levels = cs.shape[0]
			for level, (c, q) in enumerate(zip(cs, qs)):
				if level < levels:
					x = select_quadrant(x, q)
			
				x = self.GE(x, c, noise=None).detach() # G or GE?
				out.append(x)
			data[snap]['p'] = torch.stack(out)

		return data


	def get_cfg(self):

		cond_channels = len(self.config['inputs'])
		rgb_channels = len(self.config['outputs'])

		depth = self.config['depth']
		context_dim = self.config['context_dim']
		style_dim = self.config['style_dim']

		input_dim = self.config['bn_dim']
		num_layers = int(np.log2(self.config['dim'] // self.config['bn_dim'])) + 1
		nf = self.config['nf']
		filters = self.filter_setup(num_layers, nf)

		return dict_of(cond_channels, rgb_channels, depth, context_dim, style_dim,
					   input_dim, num_layers, filters)


	def filter_setup(self, num_layers, nf, nf_max=256):
		return [min(nf * 2**i, nf_max) for i in range(num_layers)]

	def log(self, key, value):
		self.logger.add_scalar(key, dget(value.mean()), self.step)

	def log_network(self, model, net):
		
		for name, value in model.named_parameters():
			if value.grad is not None:
				self.logger.add_histogram(f'{net}_grad/{name}', value.grad.cpu(), self.step)
			self.logger.add_histogram(f'{net}_params/{name}', value.cpu(), self.step)


class EMA():

	def __init__(self, beta):
		super().__init__()
		self.beta = beta

	def update_average(self, old, new):
		if old is None:
			return new
		return old * self.beta + (1 - self.beta) * new

