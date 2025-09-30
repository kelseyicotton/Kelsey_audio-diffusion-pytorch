from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
	"""Minimal feature extractor for conditioning.
	Outputs a dict of features and a fused feature tensor [B, F, T_f].
	"""

	def __init__(self, sample_rate: int, include_f0: bool = True, include_spectral: bool = True, include_noise: bool = True):
		super().__init__()
		self.sample_rate = sample_rate
		self.include_f0 = include_f0
		self.include_spectral = include_spectral
		self.include_noise = include_noise

	@torch.no_grad()
	def forward(self, waveform: Tensor) -> Dict[str, Tensor]:
		# waveform: [B, C, T] (expects C=1 or 2)
		b, c, t = waveform.shape
		mono = waveform.mean(dim=1, keepdim=True)  # [B,1,T]
		features: Dict[str, Tensor] = {}

		if self.include_f0:
			# Placeholder: energy-based proxy for pitch activity (not true F0)
			frame = 1024
			hop = 256
			pad = (frame - hop) // 2
			x = F.pad(mono, (pad, pad))
			x = x.unfold(dimension=2, size=frame, step=hop)  # [B,1,N,frame]
			energy = x.pow(2).mean(dim=-1)  # [B,1,N]
			features["f0_proxy"] = energy

		if self.include_spectral:
			# Spectral flatness proxy using simple log ratio between mean and std
			frame = 1024
			hop = 256
			pad = (frame - hop) // 2
			y = F.pad(mono, (pad, pad)).unfold(2, frame, hop)  # [B,1,N,frame]
			mag = y.abs() + 1e-6
			mean = mag.mean(dim=-1)
			std = mag.std(dim=-1) + 1e-6
			flatness = (mean / std).log()  # [B,1,N]
			features["spectral_flatness_proxy"] = flatness

		if self.include_noise:
			# Zero-crossing rate proxy
			frame = 1024
			hop = 256
			pad = (frame - hop) // 2
			z = F.pad(mono, (pad, pad)).unfold(2, frame, hop)  # [B,1,N,frame]
			sign_change = (z[..., 1:] * z[..., :-1] < 0).float().mean(dim=-1)  # [B,1,N]
			features["zcr_proxy"] = sign_change

		return features


class ConditioningAdapter(nn.Module):
	"""Projects concatenated features to context channels for UNet conditioning.
	- Input: dict of [B,1,N] features (aligned in time) or a pre-concatenated [B,F,N].
	- Output: [B, ctx_dim, N] suitable for `context_channels` at inject depth.
	"""

	def __init__(self, in_features: int, ctx_dim: int, hidden: Optional[int] = 128):
		super().__init__()
		layers = []
		if hidden:
			layers += [nn.Conv1d(in_features, hidden, kernel_size=1), nn.GELU(), nn.Conv1d(hidden, ctx_dim, kernel_size=1)]
		else:
			layers += [nn.Conv1d(in_features, ctx_dim, kernel_size=1)]
		self.proj = nn.Sequential(*layers)

	def forward(self, features: Dict[str, Tensor]) -> Tensor:
		# Concatenate along channel dimension after aligning lengths
		keys = sorted(features.keys())
		t_min = min([features[k].shape[-1] for k in keys])
		stack = [features[k][..., :t_min] for k in keys]  # [B,1,N]
		x = torch.cat(stack, dim=1)  # [B,F,N]
		return self.proj(x) 