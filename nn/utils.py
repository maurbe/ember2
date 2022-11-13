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


def make_noise(x, c=16):
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


def RAHingeLoss(r, f, mode):
	
	ra = r - f.mean()
	fa = f - r.mean()

	if mode=='D':
		return (F.relu(1 - ra) + F.relu(1 + fa)).mean()
	elif mode=='G':
		return (F.relu(1 + ra) + F.relu(1 - fa)).mean()
	

def diversityLoss(f1, f2):
	return -torch.abs((f1 - f2)).mean() # / (f2 + 1e-5)).mean()

# .......................................................................................


def add_context(x, context):
	b, _, h, w = x.shape
	c = context.shape[-1]

	c_feat = context.view(b, c, 1, 1).repeat(1, 1, h, w)
	return torch.cat([x, c_feat], dim=1)


def pair(x, y, p, context):
	x = torch.cat([x, y, p], dim=1)
	x = add_context(x, context)
	return x

"""
def pair(x, y, context):
	x = torch.cat([x, y], dim=1)
	x = add_context(x, context)
	return x
"""

def select_quadrant(p, idx, upsample=True):
	# detach, find new quadrant
	with torch.no_grad():
		x = p.detach()
		size = x.shape[-1] // 2
		x = extract_patches(x, kernel=size, stride=size)
		x = x[torch.arange(x.size(0)), idx.long()]
		if upsample:
			x = F.interpolate(x, scale_factor=2, mode='bilinear')
	return x


def extract_patches(x, kernel, stride):
	b, c, _, _ = x.shape
	windows = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
	windows = windows.permute(0, 2, 3, 1, 4, 5)
	windows = windows.reshape(b, -1, c, kernel, kernel)    
	return windows


def lowpass_resize(x):
	return F.interpolate(F.avg_pool2d(x, 2), scale_factor=2, mode='bilinear')

# .......................................................................................


def to_numpy(x):
	return x.cpu().detach().numpy()

def band_plot(ax, x, y, color, alpha=0.25, lw=1):
	low, med, up = y
	ax.fill_between(x, low, up, color=color, edgecolor='none', alpha=alpha)
	ax.plot(x, med, color=color ,lw=lw)

def band_plot_smooth(ax, x, y, color):

	npairs = len(y) // 2
	alphas = [1.0 / 2 ** i for i in range(npairs)][::-1]
	print(alphas)
	for i, alpha in zip(range(npairs), alphas):
		low, up = y[i], y[-(i+1)]
		ax.fill_between(x, low, up, color=color, edgecolor='none', alpha=alpha)

def residual_band_plot(ax, x, y1, y2, color, alpha=0.5):
	med1 = y1[1]
	low2, med2, up2 = y2
	
	rlow = (low2 - med1) / med1
	rup  = (up2  - med1) / med1
	rmed = (med2 - med1) / med1
	
	band_plot(ax, x, [rlow, rmed, rup], color=color, alpha=alpha)

# .......................................................................................

