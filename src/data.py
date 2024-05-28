import numpy as np
import torch
import glob, tqdm, random
import pyxis.torch as pxt
from torch.utils.data import Dataset, DataLoader, Sampler


class MemmapDataset(Dataset):

	def __init__(self, config, set_nr):

		axes_to_use = [('x', 'y'),
					   ('x', 'z'),
					   ('y', 'z')][set_nr]

		datapath = config['dpath']
		files = (glob.glob(datapath + f'*{axes_to_use[0]}.*.npy') + \
				 glob.glob(datapath + f'*{axes_to_use[1]}.*.npy')) #[:100]
		print('Length dataset:', len(files))
		self.len_sub = 100


		print('Warming up dataset.')
		self.all_files = []
		for f in tqdm.tqdm(files):
			self.all_files.append(np.load(f, mmap_mode='r'))
		self._update_files()

		self.batch_size = config['global_bs']
		self.frame_size = config['frame_size']

		self.maxind = self.all_files[0].shape[-1]
		self.step = 0
		self.num_steps = self.__len__() // self.batch_size - 1


	def _update_files(self):
		# choose a random subset
		random.shuffle(self.all_files)
		self.files = self.all_files[:self.len_sub]


	def __len__(self):
		# expand the length of the total ds
		return self.len_sub * (self.maxind // self.frame_size) ** 2


	def __getitem__(self, index):

		xl = np.random.randint(0, self.maxind - self.frame_size); xr = xl + self.frame_size
		yl = np.random.randint(0, self.maxind - self.frame_size); yr = yl + self.frame_size

		random_idx = np.random.randint(0, self.len_sub)
		file = self.files[random_idx]

		x = file[0, :2,  xl:xr, yl:yr]
		y = file[0, 2:6, xl:xr, yl:yr]
		c = file[0, 6:,  0, 0]
		q = np.ones((1,))

		self.step += 1
		if self.step > self.num_steps:
			self.step = 0
			self._update_files()

		return {'x': torch.from_numpy(x),
				'y': torch.from_numpy(y),
				'c': torch.from_numpy(c),
				'q': torch.from_numpy(q)
				}


def load_sample(datapath, snaps, axis, slc, device):

	data = {}
	for snap in snaps:
		d = np.load(datapath + f'fbox_snapshot_{snap}.{axis}.{slc}.npy')
		data[f'{snap}'] = torch.from_numpy(d).to(device)

	return data
