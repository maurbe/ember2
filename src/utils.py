import torch
import torch.nn.functional as F
import numpy as np
import random
import json
from configparser import RawConfigParser


# .......................................................................................


def load_config(cfg):

	parser = RawConfigParser()
	parser.read(cfg)

	h = {s:dict(parser.items(s)) for s in parser.sections()}
	d = {}
	for key in h['config'].keys():
		d[key] = json.loads(parser.get('config', key))
	return d

# .......................................................................................


def set_seed(seed):
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)


def summary(net):
	return r'{}e6'.format(round(sum(p.numel() for p in net.parameters()) / 1e6, 1))


def dget(tensor):
	return tensor.detach().item()


def to_numpy(x):
	return x.cpu().detach().numpy()


def make_noise(x, c=8):
		n, _, h, w = x.shape # promoted c to kwarg
		return torch.normal(0, 0.1, (n, c, h, w), device=x.device)


def toggle_grad(nets, requires_grad=False):
	"""Set requies_grad=False for all the networks to avoid unnecessary computations
	Parameters:
		nets (network list)   -- a list of networks
		requires_grad (bool)  -- whether the networks require gradients or not
	"""
	if not isinstance(nets, list):
		nets = [nets]
	for net in nets:
		if net is not None:
			for param in net.parameters():
				param.requires_grad = requires_grad


def transfer_to(data, device):
		for key in data.keys():
			if type(data[key]) is dict:
				for subkey in data[key].keys():
					data[key][subkey] = data[key][subkey].to(device)
			else:
				data[key] = data[key].to(device)
		return data

# .......................................................................................


def NSLoss(r, f, mode):

	if mode=='D':
		return (F.softplus(-r) + F.softplus(f)).mean()
	elif mode=='G':
		return F.softplus(-f).mean()

# .......................................................................................


def add_context(x, context):
	b, _, h, w = x.shape
	c = context.shape[-1]

	c_feat = context.view(b, c, 1, 1).repeat(1, 1, h, w)
	return torch.cat([x, c_feat], dim=1)


def pair_base(x, y, p, context):
	x = torch.cat([x, y, p], dim=1)
	x = add_context(x, context)
	return x

# .......................................................................................
