import torch
import torch.nn as nn
import torch.nn.functional as F
import os, glob, tqdm, numpy as np

from torch.optim import Adam
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
		cfg = self.get_cfg(mode)

		self.G = Generator(mode, cfg)
		self.D = Discriminator(mode, **cfg)

		self.GE  = Generator(mode, cfg)
		self.ema = EMA(0.995)
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


	def configure_optimizers(self):

		lr_g = self.config['lr_g']
		lr_d = self.config['lr_d']
		self.opt_g = Adam(self.G.parameters(), lr=lr_g, betas=(0.5, 0.9)) # base: (0.0, 0.99)
		self.opt_d = Adam(self.D.parameters(), lr=lr_d, betas=(0.5, 0.9)) # base: (0.0, 0.99)


	def qim(self, x, cmap='viridis'):
		import matplotlib.pyplot as plt
		plt.figure()
		plt.imshow(x[0].detach().cpu().numpy(), cmap=cmap)
		plt.show()


	def _step_base(self, x, y, c):

		# train G
		self.opt_g.zero_grad(set_to_none=True)
		toggle_grad(self.G, True)
		toggle_grad(self.D, False)

		p1 = self.G(x, c)
		p2 = self.G(x, c)

		f = self.D(pair_base(x, p1, p2, c))
		g_loss = NSLoss(None, f, mode='G')
		g_loss.backward()
		self.opt_g.step()


		# train D
		self.opt_d.zero_grad(set_to_none=True)
		toggle_grad(self.D, True)
		toggle_grad(self.G, False)

		p1 = p1.detach()
		p2 = p2.detach()

		r = self.D(pair_base(x, y,  p1, c))
		f = self.D(pair_base(x, p1, p2, c))

		d_loss = NSLoss(r, f, mode='D')
		d_loss.backward()
		self.opt_d.step()

		self.log(f'g_adv', g_loss)
		self.log(f'dfake', f)
		self.log(f'dtrue', r)
		self.log(f'd_adv', d_loss)

		return p1



	def train_step_base(self, batch):

		x = batch['x']
		y = batch['y']#[:, 0] # lvl 0
		c = batch['c']#[:, 0] # lvl 0
		p =  self._step_base(x, y, c)



	def train_step(self, step, batch, logger):
		self.step = step
		self.logger = logger

		self.train_step_base(batch)

		if self.step % 10 == 0 and self.step > 50000:
			self.EMA()

		if self.step % 50000 == 0:
			self.log_network(self.G, "G")
			self.log_network(self.D, "D")

		self.logger = None


	def predict_base(self, data):
		self.eval()

		for snap in tqdm.tqdm(data.keys()):
			x = data[snap]['x']
			c = data[snap]['c'][0] # lvl 0

			data[snap]['y'] = data[snap]['y'][0]
			data[snap]['p'] = self.GE(x, c, noise=None).detach() # G or GE?
		return data


	def discriminate_base(self, data):
		self.eval()

		for snap in tqdm.tqdm(data.keys()):

			x = data[snap]['x']
			y = data[snap]['y'][0]
			c = data[snap]['c'][0]

			p1 = self.GE(x, c, noise=None).detach()
			p2 = self.GE(x, c, noise=None).detach()

			data[snap]['fakes'] = self.D(pair_base(x, p1, p2, c)).detach()
			data[snap]['reals'] = self.D(pair_base(x, y , p2, c)).detach()

			# free memory
			del data[snap]['x']
			del data[snap]['y']
			del data[snap]['c']

		return data


	def get_cfg(self, mode):

		inp_channels = len(self.config['inputs'])
		if mode=='zoom':
			inp_channels = len(self.config['outputs'])
		out_channels = len(self.config['outputs'])

		context_dim = self.config['context_dim']
		style_dim 	= self.config['style_dim']
		style_depth = self.config['style_depth']

		nl = self.config['nl']
		nf = self.config['nf']
		filters = self.filter_setup(nl, nf)

		return dict_of(inp_channels,
					   out_channels,
					   context_dim,
					   style_dim,
					   style_depth,
					   nl,
					   filters)


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
