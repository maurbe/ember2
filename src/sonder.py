import torch
import torch.nn as nn

# seems to boost speed a bit...
torch.backends.cudnn.enabled=True

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


	def load_ckpt(self, set_nr, mode, ckpt=None):

		pdir = f'{self.path}/models/runs/{self.run_name}/{mode}/set_{set_nr}.'
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
		self.nn_zoom.aux.eval()
		toggle_grad(self.nn_zoom.aux, False)
		print('Assigned aux!')


	def train(self, set_nr, mode):

		#train_ds = datatrain(self.config, set_nr) #
		dataloader = DataLoader(dataset=MemmapDataset(self.config, set_nr),
								batch_size=self.config['global_bs'],
								num_workers=os.cpu_count())

		logger = SummaryWriter(log_dir=f'{self.run_dir}/{mode}', filename_suffix=f'.set_{set_nr}.{self.device}')
		nn = self.nn_base if mode=='base' else self.nn_zoom
		nn.to(self.device)

		step = 0
		for epoch in range(10000):
			print('Epoch:', epoch, flush=True)
			for batch in tqdm.tqdm(dataloader): # train_ds

				nn.train_step(step,
							  transfer_to(batch, self.device),
							  logger)

				if (step % self.ckpt_freq == 0):
					torch.save(nn, self.run_dir + f'/{mode}/set_{set_nr}.ckpt_{step}.pt')
				step += 1
