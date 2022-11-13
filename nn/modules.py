import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.nn.layers import *
from lib.nn.utils import *


class Generator(nn.Module):

	def __init__(self, config):
		super().__init__()
		
		self.mapping = MappingNet(**config)
		self.synthesis = SynthesisNet(**config)

	def forward(self, x, context, noise=None):
		style = self.mapping(context) #, latent)
		out = self.synthesis(x, style, noise)
		return out


class MappingNet(nn.Module):

	def __init__(self, depth, style_dim, context_dim, **kwargs):
		super().__init__()

		layers = []
		for i in range(depth):
			layers.append(nn.Linear(context_dim if i==0 else style_dim, style_dim))
			layers.append(nn.LeakyReLU(0.1))
		self.net = nn.Sequential(*layers)

	def forward(self, context):
		return self.net(context)


class SynthesisNet(nn.Module):

	def __init__(self, 
				 style_dim, 
				 cond_channels, 
				 rgb_channels, 
				 filters,
				 **kwargs):
		super().__init__()
		
		self.dblocks = nn.ModuleList()
		self.ublocks = nn.ModuleList()

		fixed_channels = 32
		for i in range(len(filters)):

			is_first = (i == 0)
			is_last  = (i == len(filters) - 1)

			dblock = dBlock(
							inp_ch=fixed_channels if not is_first else cond_channels,
							out_ch=fixed_channels,
							is_last=is_last)

			ublock = SynthesisBlock(
							style_dim=style_dim,
							inp_ch=filters[i],
							out_ch=filters[max(i - 1, 0)],
							rgb_ch=rgb_channels,
							cond_ch=fixed_channels,
							is_last=is_last,
							is_first=is_first)
			
			self.dblocks.append(dblock)
			self.ublocks.append(ublock)


	def forward(self, x, style, noise):

		ys = []
		for block in self.dblocks:
			x, y = block(x)
			ys.append(y)

		x = None
		for block, y in zip(self.ublocks[::-1], ys[::-1]):
			x = block(x, y, style, noise)
		return x


class Discriminator(nn.Module):

	def __init__(self, 
			     cond_channels, 
			     context_dim, 
			     rgb_channels, 
			     filters, 
			     **kwargs):
		super().__init__()

		# rgb_channels * 2 for Adler approach
		in_channels = cond_channels + rgb_channels * 2 + context_dim
		filters = [in_channels] + filters

		self.blocks = nn.ModuleList()
		for i in range(len(filters) - 1):

			is_last  = (i == len(filters) - 2)
			if not is_last:
				block = DiscriminatorBlock(inp_ch=filters[i],
										   out_ch=filters[i+1])

			else:
				block = FinalDiscriminatorBlock(filters[i])
			self.blocks.append(block)


	def forward(self, x):

		for i, block in enumerate(self.blocks):
			x = block(x)
		return x

