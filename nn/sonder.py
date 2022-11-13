import torch
import torch.nn as nn

import numpy as np
import tqdm
import os, glob
from pathlib import Path

from lib.nn.net import *
from lib.nn.utils import *
from lib.nn.data import *
from torch.utils.tensorboard import SummaryWriter

# ........................................................................


class sonder():

	def __init__(self, cfg):
		
		self.solve_env(cfg)	
		self.nn_base = net(self.config, mode='base')
		self.nn_zoom = net(self.config, mode='zoom')
		self.test_sampler = datatest(self.config)


	def solve_env(self, cfg):

		self.path 		= str(Path(os.path.dirname(__file__)).parent.parent)
		self.run_name   = cfg.split('.')[0]
		self.config_dir = self.path + '/models/configs/' + cfg
	
		self.run_dir    = self.path + '/models/runs/' + self.run_name
		self.config 	= load_config(self.config_dir)
		self.ckpt_freq	= self.config['ckpt_freq']

		paths = [self.run_dir + '/base', 
				 self.run_dir + '/zoom']
		for path in paths:				
			os.makedirs(path, exist_ok=True)

		if torch.cuda.is_available():
			self.device = torch.device("cuda:0")
		else:
			self.device = torch.device("cpu")
		print(self.device)
		set_seed(1)


	def load_ckpt(self, mode, ckpt=None):

		pdir = f'{self.path}/models/runs/{self.run_name}/{mode}/'
		if ckpt is None:
			ckpt = max(glob.glob(pdir + 'ckpt_*.pt'), 
					   key=os.path.getctime).split('ckpt_')[-1].strip('.pt')
		path = pdir + f'ckpt_{ckpt}.pt'
		
		if mode=='base':
			self.nn_base = torch.load(path, map_location=self.device)
		else:
			self.nn_zoom = torch.load(path, map_location=self.device)
		print(f'ckpt {mode}: {ckpt}')


	def assign_aux(self):
		self.nn_zoom.aux = self.nn_base.GE
		print('Assigned aux!')


	def train(self, mode):

		train_ds = datatrain(mode, self.config)
		logger = SummaryWriter(log_dir=f'{self.run_dir}/{mode}', 
							   filename_suffix=f'.{self.device}')
		nn = self.nn_base if mode=='base' else self.nn_zoom
		nn.to(self.device)
		
		for step in tqdm.trange(100000000):
			batch = transfer_to(next(train_ds), self.device)
			nn.train_step(step, batch, logger)

			if (step % self.ckpt_freq == 0):
				torch.save(nn, self.run_dir + f'/{mode}/ckpt_{step}.pt')


	def predict(self, mode, discriminate=False, **kwargs):

		self.nn_base.eval().to(self.device)
		self.nn_zoom.eval().to(self.device)
		
		data = self.test_sampler.yield_sample(**kwargs, mode=mode)
		data = transfer_to(data, self.device)

		with torch.no_grad():
			if not discriminate:
				if mode=='base':
					return self.nn_base.predict_base(data)
				else:
					return self.nn_zoom.predict_zoom(data)
			else:
				return self.nn_base.discriminate_base(data)

	"""
	def permute_data(self, mode, patch_sizes, **kwargs):

		self.nn_base.eval().to(self.device)
		self.engine = Engine(self.nn_base)

		data = self.test_sampler.yield_sample(**kwargs, mode=mode)
		data = transfer_to(data, self.device)
		
		return self.engine.create_permuation_sets(data, patch_sizes)
	"""

