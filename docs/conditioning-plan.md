# Conditioning Plan (PEFT Baseline)

## Goals
- Add parameter-efficient conditioning to existing diffusion UNet.
- Start with bottleneck-only conditioning; keep base UNet frozen.
- Support features relevant to extended vocal techniques (F0, spectral/noise proxies).

## Inject Strategy
- Use `UNetV0` `context_channels` to inject conditioning at the bottleneck (single depth initially).
- `inject_depth`: bottleneck layer index (to be confirmed after instantiation based on `channels` length).
- Future: expand to multi-depth conditioning if needed.

## Feature Set (v0)
- F0 contour (CREPE/YIN or similar), normalized and interpolated to bottleneck length.
- Spectral irregularity proxy (e.g., spectral flatness/centroid variation, HNR proxy).
- Noise proxy (e.g., zero-crossing rate, high-band energy ratio).

## Adapter (Trainable)
- `ConditioningAdapter`: small MLP that maps concatenated features per time step to `ctx_dim`.
- Optionally FiLM-style scaling if later used at multiple depths.
- PEFT-lite: train adapter only; freeze base UNet parameters.

## Data Flow
- Dataset yields waveform `[B, C, T]`.
- `FeatureExtractor` computes per-example feature tensor `[B, F, T_f]`.
- Interpolate to bottleneck time-length `[B, F, T_b]`.
- Adapter projects to context channels `[B, ctx_dim, T_b]`.
- Pass as `channels=[None, ..., ctx, ..., None]` to diffusion net.

## Minimal Changes
- New module: `audio_diffusion_pytorch/conditioning.py` with `FeatureExtractor`, `ConditioningAdapter`.
- Training hook: create adapter, freeze base UNet, optimize adapter params.

## Open Decisions
- Exact `ctx_dim` (start: 32 or 64).
- Chosen F0 extractor (CREPE small vs YIN), compute-time budget.
- `inject_depth` index (depends on UNet depth used in config).

## Immediate Next Steps
1. Implement `FeatureExtractor` (F0 + simple spectral/noise features).
2. Implement `ConditioningAdapter` (MLP) and bottleneck wiring.
3. Add training flag to freeze base UNet and train adapter only.

## Implementation notes (this branch)
- Bottleneck context wiring:
  - Set `inject_depth = len(channels) // 2` and `ctx_dim = 32`.
  - Built `context_channels` with `ctx_dim` at `inject_depth`; replaced the model’s UNet with a context-aware `UNetV0`.
  - Synchronized `self.model.diffusion.net` and `self.model.sampler.net` to the same UNet.
- Adapter and features:
  - Added `audio_diffusion_pytorch/conditioning.py` with `FeatureExtractor` (proxy features) and `ConditioningAdapter`.
  - In training, compute features → adapter → interpolate to bottleneck length → pass via `channels` only at `inject_depth`.
- PEFT-lite freeze and optimizer:
  - Froze base UNet parameters; trained only the adapter (confirmed Base trainable: 0; Adapter trainable: ~4.6k).
  - Optimizer set to `AdamW` over adapter params only.
- Sampling fix:
  - During monitoring, provided a zero context tensor at the bottleneck (`[B, ctx_dim, T_b]`) so sampling works with the context-aware UNet.
- Logs and verification:
  - Added prints to confirm inject depth, ctx dim, and trainable parameter counts after freezing. 