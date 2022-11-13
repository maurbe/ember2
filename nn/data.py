import numpy as np
import torch
import pyxis.torch as pxt

from torch.utils.data import Dataset, DataLoader, Sampler
from lib.nn.torch_utils.misc import InfiniteSampler


def datatrain(mode, config):

	data_path = config['dpath']
	ds = pxt.TorchDataset(data_path + f'{mode}_ds')
	sampler = InfiniteSampler(dataset=ds)
	
	return iter(DataLoader(dataset=ds,
					  	   sampler=sampler,
					  	   batch_size=config['global_bs'], 
					  	   num_workers=config['global_bs'],
					  	   pin_memory=True))


class datatest():
	
	def __init__(self, config):
		self.dp = config['dpath']
	
	def yield_sample(self, mode, sim, snaps, axis, slc, q):

		data = {}
		for snap in snaps:

			out = {}
			if axis is not None:
				db = pxt.Reader(dirpath=self.dp + f'global/nn_{mode}_{sim}_{snap}_{axis}_{slc}')
			else:
				db = pxt.Reader(dirpath=self.dp + f'global/nn_{mode}_{sim}_{snap}_{slc}')
			sample = db[q]
			db.close()

			if mode=='zoom':
				out['x'] = torch.from_numpy(sample['x'][:, np.newaxis])
				out['y'] = torch.from_numpy(sample['y'][:, np.newaxis])
				out['c'] = torch.from_numpy(sample['c'][:, np.newaxis])
				out['q'] = torch.from_numpy(sample['q'])

			else:
				out['x'] = torch.from_numpy(sample['x'][np.newaxis])
				out['y'] = torch.from_numpy(sample['y'][np.newaxis])
				out['c'] = torch.from_numpy(sample['c'][np.newaxis])

			data[f'{snap}'] = out
		return data

