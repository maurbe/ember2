import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.nn.layers import *
from lib.nn.utils import *


class Generator(nn.Module):

	def __init__(self, mode, config):
		super().__init__()

		self.mapping = MappingNet(**config)
		self.synthesis = SynthesisNet(mode, **config)

	def forward(self, x, context, noise=None):
		style = self.mapping(context)
		out = self.synthesis(x, style, noise)
		return out


class MappingNet(nn.Module):

	def __init__(self, style_depth, style_dim, context_dim, **kwargs):
		super().__init__()

		layers = []
		for i in range(style_depth):
			layers.append(nn.Linear(context_dim if i==0 else style_dim, style_dim))
			layers.append(nn.LeakyReLU(0.1))
		self.net = nn.Sequential(*layers)

	def forward(self, context):
		return self.net(context)


class SynthesisNet(nn.Module):

	def __init__(self,
				 mode,
				 style_dim,
				 inp_channels,
				 out_channels,
				 filters,
				 **kwargs):
		super().__init__()

		self.mode = mode
		self.dblocks = nn.ModuleList()
		self.ublocks = nn.ModuleList()

		fixed_channels = filters[0]
		for i in range(len(filters)):

			is_first = (i == 0)
			is_last  = (i == len(filters) - 1)

			dblock = dBlock(
							inp_ch=fixed_channels if not is_first else inp_channels,
							out_ch=fixed_channels,
							is_last=is_last)

			ublock = SynthesisBlock(
							style_dim=style_dim,
							inp_ch=filters[i],
							out_ch=filters[max(i - 1, 0)],
							rgb_ch=out_channels,
							cond_ch=fixed_channels,
							is_last=is_last,
							is_first=is_first)

			self.dblocks.append(dblock)
			self.ublocks.append(ublock)

		if self.mode=='zoom':
			self.cl = ConstraintLayer()


	def forward(self, x, style, noise):
		xres = x

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
				 mode,
			     inp_channels,
			     out_channels,
			     context_dim,
			     filters,
			     **kwargs):
		super().__init__()

		if mode=='base':
			in_channels = inp_channels + out_channels * 2 + context_dim # out_channels * 2 for Adler approach
		elif mode=='zoom':
			#in_channels = inp_channels + out_channels * 2 + context_dim # +0 -> *2
			in_channels = (inp_channels + out_channels) * 2 + context_dim
		else:
			raise SystemExit

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
