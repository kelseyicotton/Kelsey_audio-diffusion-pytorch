from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
	"""Feature extractor for conditioning.
	Outputs a dict of features with shape [B, 1, T_f] each (same frame grid).

	Features (v1):
	- f0_yin (Hz, normalized to [0,1] range by fmin-fmax) if torchaudio YIN is available, otherwise energy proxy
	- spectral_centroid (Hz, normalized by Nyquist)
	- spectral_flatness (log-flatness)
	- zero_crossing_rate

	Config:
	- sample_rate: audio sampling rate
	- frame_length: analysis window length in samples
	- hop_length: hop size in samples
	- fmin/fmax: F0 range for YIN
	"""

	def __init__(
		self,
		sample_rate: int,
		frame_length: int = 1024,
		hop_length: int = 256,
		fmin: float = 50.0,
		fmax: float = 1000.0,
		include_f0: bool = True,
		include_spectral: bool = True,
		include_noise: bool = True,
	):
		super().__init__()
		self.sample_rate = sample_rate
		self.frame_length = frame_length
		self.hop_length = hop_length
		self.fmin = fmin
		self.fmax = fmax
		self.include_f0 = include_f0
		self.include_spectral = include_spectral
		self.include_noise = include_noise

	@torch.no_grad()
	def forward(self, waveform: Tensor) -> Dict[str, Tensor]:
		# waveform: [B, C, T]
		b, c, t = waveform.shape
		device = waveform.device
		dtype = waveform.dtype
		mono = waveform.mean(dim=1, keepdim=False)  # [B,T]
		features: Dict[str, Tensor] = {}

		# Compute number of frames for framing-based features
		# Center padding to align with STFT conventions
		pad = (self.frame_length - self.hop_length) // 2
		mono_pad = F.pad(mono.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)  # [B,Tp]
		# Frame with unfold for ZCR and energy fallbacks
		frames = mono_pad.unfold(dimension=1, size=self.frame_length, step=self.hop_length)  # [B, N, frame]
		N = frames.shape[1]

		# 1) F0 using torchaudio YIN if available
		if self.include_f0:
			f0_feat: Optional[Tensor] = None
			try:
				import torchaudio.functional as AF  # type: ignore
				# torchaudio Yin expects [B, T]; frame_time ~ hop/sample_rate
				frame_time = self.hop_length / float(self.sample_rate)
				win_length = self.frame_length
				f0 = AF.detect_pitch_frequency(
					mono, sample_rate=self.sample_rate, frame_time=frame_time, win_length=win_length, freq_low=self.fmin, freq_high=self.fmax
				)  # [B, N]
				# Replace NaNs/zeros with 0
				f0 = torch.nan_to_num(f0, nan=0.0)
				# Normalize to [0,1] by fmin-fmax
				norm = (f0 - self.fmin) / max(self.fmax - self.fmin, 1e-6)
				norm = norm.clamp(0.0, 1.0)
				f0_feat = norm.unsqueeze(1)  # [B,1,N]
			except Exception:
				# Fallback energy proxy per frame
				energy = (frames.pow(2).mean(dim=-1)).unsqueeze(1)  # [B,1,N]
				# Normalize energy per-batch for stability
				max_e = energy.amax(dim=-1, keepdim=True).clamp_min(1e-6)
				f0_feat = (energy / max_e)
			features["f0"] = f0_feat

		# 2) Spectral features via STFT magnitude
		if self.include_spectral:
			# STFT
			stft = torch.stft(
				mono,
				n_fft=self.frame_length,
				hop_length=self.hop_length,
				win_length=self.frame_length,
				window=torch.hann_window(self.frame_length, device=device, dtype=dtype),
				center=True,
				return_complex=True,
			)
			# [B, F, N]
			mag = stft.abs().clamp_min(1e-8)
			B, Freq, N_stft = mag.shape
			# Match N (framing) and STFT frames if off by 1
			N_common = min(N, N_stft)
			mag = mag[..., :N_common]
			if "f0" in features:
				features["f0"] = features["f0"][..., :N_common]
			# Frequencies in Hz
			freqs = torch.linspace(0.0, self.sample_rate / 2.0, Freq, device=device, dtype=dtype).view(1, Freq, 1)
			power = mag
			# Spectral centroid (Hz)
			centroid = (power * freqs).sum(dim=1) / power.sum(dim=1).clamp_min(1e-8)  # [B,Nc]
			centroid_norm = (centroid / (self.sample_rate * 0.5)).clamp(0.0, 1.0).unsqueeze(1)  # [B,1,Nc]
			# Spectral flatness (log of geometric / arithmetic mean)
			geo = torch.exp(torch.log(power).mean(dim=1))
			ari = power.mean(dim=1).clamp_min(1e-8)
			flatness = (geo / ari).clamp_min(1e-8).log().unsqueeze(1)  # [B,1,Nc]
			features["spectral_centroid"] = centroid_norm
			features["spectral_flatness"] = flatness

		# 3) Noise proxy: zero-crossing rate per frame
		if self.include_noise:
			z = frames[..., :N_common, :]
			sign_change = (z[..., 1:] * z[..., :-1] < 0).float().mean(dim=-1)  # [B,Nc]
			features["zcr"] = sign_change.unsqueeze(1)

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