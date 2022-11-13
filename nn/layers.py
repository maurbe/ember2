import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from lib.nn.torch_utils import misc
from lib.nn.torch_utils import persistence
from lib.nn.torch_utils.ops import conv2d_resample
from lib.nn.torch_utils.ops import upfirdn2d
from lib.nn.torch_utils.ops import bias_act
from lib.nn.torch_utils.ops import fma


@misc.profiled_function
def modulated_conv2d(
	x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
	weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
	styles,                     # Modulation coefficients of shape [batch_size, in_channels].
	noise           = None,     # Optional noise tensor to add to the output activations.
	up              = 1,        # Integer upsampling factor.
	down            = 1,        # Integer downsampling factor.
	padding         = 0,        # Padding with respect to the upsampled image.
	resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
	demodulate      = True,     # Apply weight demodulation?
	flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
	fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
	):
	batch_size = x.shape[0]
	out_channels, in_channels, kh, kw = weight.shape
	misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
	misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
	#misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

	# Pre-normalize inputs to avoid FP16 overflow.
	"""
	if x.dtype == torch.float16 and demodulate:
		weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
		styles = styles / styles.norm(float('inf'), dim=1, keepdim=True)
	"""

	# Calculate per-sample weights and demodulation coefficients.
	w = None
	dcoefs = None
	if demodulate or fused_modconv:
		w = weight.unsqueeze(0) # [NOIkk]
		w = w * styles.reshape(batch_size, 1, -1, 3, 3) # 1, 1) # [NOIkk]
	if demodulate:
		dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
	if demodulate and fused_modconv:
		w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

	# Execute by scaling the activations before and after the convolution.
	if not fused_modconv:
		x = x * styles.to(x.dtype).reshape(batch_size, -1, 3, 3) #1, 1)
		x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
		if demodulate and noise is not None:
			x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
		elif demodulate:
			x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
		elif noise is not None:
			x = x.add_(noise.to(x.dtype))
		return x

	# Execute as one fused op using grouped convolution.
	with misc.suppress_tracer_warnings(): # this value will be treated as a constant
		batch_size = int(batch_size)
	misc.assert_shape(x, [batch_size, in_channels, None, None])
	x = x.reshape(1, -1, *x.shape[2:])
	w = w.reshape(-1, in_channels, kh, kw)
	x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
	x = x.reshape(batch_size, -1, *x.shape[2:])
	if noise is not None:
		x = x.add_(noise)
	return x


# Lightweight ModConv2d Layer implementation: 
# -> bias_act was removed entirely
class ModulatedConv2d(nn.Module):

	def __init__(
		self,
		style_dim,
		in_channel,
		out_channel,

		kernel_size=3,
		demodulate=True,
		upsample=True,
		resample_filter=[1, 3, 3, 1],
		fused_modconv=True):
		super().__init__()

		self.up = upsample
		self.fused_modconv = fused_modconv
		self.padding = kernel_size // 2

		self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
		self.weight = nn.Parameter(th.randn(out_channel, in_channel, kernel_size, kernel_size))

		# Full modulation
		self.affine = nn.Linear(style_dim, in_channel * kernel_size ** 2)
		self.act = nn.LeakyReLU(0.2)
		self.bias = nn.Parameter(torch.zeros([out_channel, 1, 1]))


	def forward(self, x, w, gain=1, noise=None):
		
		b, c, h, _ = x.shape
		styles = self.affine(w)

		# -> ToDo: WHAT DOES THIS DO: correlation instead of convolution -> same in the end
		flip_weight = (self.up == 1) # slightly faster
		x = modulated_conv2d(x=x, 
							 weight=self.weight, 
							 styles=styles, 
							 noise=noise, 
							 up=self.up,
							 padding=self.padding, 
							 resample_filter=self.resample_filter, 
							 flip_weight=flip_weight, 
							 fused_modconv=self.fused_modconv)
		return self.act(x + self.bias)


class dBlock(nn.Module):

	def __init__(self, 
				 inp_ch,
				 out_ch,
				 is_last
				 ):
		super().__init__()

		self.conv1 = nn.Conv2d(inp_ch, out_ch, 3, 1, 1)
		self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
		self.act   = nn.LeakyReLU(0.2, inplace=True)

		self.is_last = is_last
		if not is_last:
			self.pool = nn.Conv2d(out_ch, out_ch, 3, 2, 1)

	def forward(self, x):

		x = self.act(self.conv1(x))
		x = self.act(self.conv2(x))
		y = x
		
		if not self.is_last:
			x = self.pool(x)
		return x, y


class SynthesisBlock(nn.Module):

	def __init__(self, 
				 style_dim, 
				 inp_ch, 
				 out_ch, 
				 rgb_ch, 
				 cond_ch,
				 is_last,
				 is_first):
		super().__init__()

		self.noise_ch = 8
		total_ch = cond_ch if is_last else inp_ch + cond_ch
		
		self.conv1 = ModulatedConv2d(style_dim, total_ch, out_ch)
		self.conv2 = ModulatedConv2d(style_dim, out_ch + self.noise_ch, out_ch)

		self.is_first = is_first
		if self.is_first:
			self.iconv  = ModulatedConv2d(style_dim, out_ch, rgb_ch)

	
	def cat_noise(self, x, noise=None):
		b, c, h, w = x.shape
		if noise is None:
			n = x.new_empty(*x.shape).normal_(std=0.1)[:, :self.noise_ch]
		else:
			n = noise[:, :self.noise_ch, :h, :w]
		return th.cat([x, n], dim=1)
	

	def forward(self, x, y, style, noise=None):
		
		if x is not None:
			x = F.interpolate(x, scale_factor=2, mode='bilinear')
			x = th.cat([x, y], dim=1)
		else:
			x = y
		
		x = self.conv1(x, style)
		x = self.cat_noise(x, noise)
		x = self.conv2(x, style)

		if self.is_first:
			x = self.iconv(x, style)
		return x
 

class DiscriminatorBlock(nn.Module):

	def __init__(self, inp_ch, out_ch):
		super().__init__()

		# left path
		self.skip  = nn.Conv2d(inp_ch, out_ch, 3, 2, 1)
		self.conv1 = nn.Conv2d(inp_ch, out_ch, 3, 1, 1) 
		self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1) 
		
		self.pool  = nn.Conv2d(out_ch, out_ch, 3, 2, 1)
		self.act   = nn.LeakyReLU(0.2, inplace=True)

		# right path
		self.fftc1 = nn.Conv2d(inp_ch * 2, out_ch * 2, 3, 1, 1)
		self.fftc2 = nn.Conv2d(out_ch * 2, out_ch * 2, 3, 1, 1)


	def forward(self, x):
		x_ = x 
		skip = self.skip(x)

		# left path
		x = self.act(self.conv1(x))
		x = self.act(self.conv2(x)) #+ f

		# right path
		y = th.fft.fft2(x_)
		y = th.cat([y.real, y.imag], dim=1)
		y = self.act(self.fftc1(y))
		y = self.act(self.fftc2(y))

		yr, yi = torch.chunk(y, 2, dim=1)
		y = th.complex(yr, yi)
		y = th.fft.ifft2(y).real

		# residual and pool
		x = x + y
		x = self.pool(x) 
		return x + skip

"""
class FourierFeatures(nn.Module):

	def __init__(self, out_channel):
		super().__init__()
		
		self.conv1 = nn.Conv2d(out_channel * 2, out_channel * 2, 1, 1)
		self.conv2 = nn.Conv2d(out_channel * 2, out_channel * 2, 1, 1)
		self.act   = nn.LeakyReLU(0.2, inplace=True)

	def forward(self, x):
		_, _, h, w = x.shape
		
		y = torch.fft.rfft2(x, norm='backward')
		y = torch.cat([y.real, y.imag], dim=1)

		# DeepRFT has a second 1x1 convolution
		y = self.conv2(self.act(self.conv1(y))) 

		y_real, y_imag = torch.chunk(y, 2, dim=1)
		y = torch.complex(y_real, y_imag)

		y = torch.fft.irfft2(y, s=(h, w), norm='backward')
		return y
"""

class FinalDiscriminatorBlock(nn.Module):

	def __init__(self, inp_ch):
		super().__init__()

		self.conv = nn.Conv2d(inp_ch, inp_ch, 3, 1, 1)
		self.act  = nn.LeakyReLU(0.2, inplace=True)
		self.fc = nn.Conv2d(inp_ch, 1, 4)

	def flatten(self, x):
		return x.reshape(x.shape[0], -1)

	def forward(self, x):

		x = self.act(self.conv(x))
		x = self.flatten(self.fc(x))
		return x


# ...............................................................
"""
class SynthesisBlock(nn.Module):

	def __init__(self, 
				 style_dim, 
				 in_channel, 
				 out_channel, 
				 rgb_channel, 
				 cond_channel,
				 is_last,
				 is_first):
		super().__init__()

		self.fourier_channels = 16
		self.noise_channels = 8 #!!!!!!!!
		total_channels = cond_channel if is_last else in_channel + cond_channel
		
		self.ffg = FFG(style_dim,
					   total_channels, 
					   self.fourier_channels)

		self.conv1 = ModulatedConv2d(style_dim, 
									 total_channels, 
									 out_channel) # + self.noise_channels
		self.conv2 = ModulatedConv2d(style_dim, 
									 out_channel + self.fourier_channels + self.noise_channels, 
									 out_channel)

		self.is_first = is_first
		if self.is_first:
			self.iconv  = ModulatedConv2d(style_dim, 
										  out_channel, 
										  rgb_channel)

	
	def cat_noise(self, x, noise=None):
		b, c, h, w = x.shape
		if noise is None:
			n = x.new_empty(*x.shape).normal_(std=0.1)[:, :self.noise_channels] # !!!!!!!!!!
		else:
			n = noise[:, :self.noise_channels, :h, :w]
		return th.cat([x, n], dim=1)
	

	def forward(self, x, y, style, noise=None):
		
		if x is not None:
			x = F.interpolate(x, scale_factor=2, mode='bilinear')
			x = th.cat([x, y], dim=1)
		else:
			x = y
		
		f = self.ffg(x, style)
		x = self.conv1(x, style)
		x = th.cat([x, f], dim=1)

		x = self.cat_noise(x, noise)
		x = self.conv2(x, style)

		if self.is_first:
			x = self.iconv(x, style)
		return x
class FFG(nn.Module):

	def __init__(self, style_dim, in_channel, fourier_channel):
		super().__init__()
		
		self.reduce_conv = ModulatedConv2d(style_dim,
										   in_channel, 
									 	   fourier_channel)
		self.fconv = ModulatedConv2d(style_dim, 
									 fourier_channel * 2, 
									 fourier_channel * 2)
		self.act = nn.LeakyReLU(0.2, inplace=True)

	def forward(self, x, style):

		x = self.act(self.reduce_conv(x, style))
		x = th.fft.fft2(x)
		x = th.cat([x.real, x.imag], dim=1)

		x = self.act(self.fconv(x, style))
		xr, xi = th.chunk(x, 2, dim=1)
		x = torch.complex(xr, xi)

		x = th.fft.ifft2(x).real
		return x
"""