#!/usr/bin/env python3
"""
Simple audio generation script with robust WAV file saving
Works on Windows without additional audio backend dependencies.
Now supports loading a trained model checkpoint.
"""

import torch
import numpy as np
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import os
import argparse
import time


def save_audio_as_wav(audio_tensor, filename, sample_rate=16000):
    """Save audio tensor as WAV file using scipy (more reliable on Windows)."""
    try:
        import scipy.io.wavfile as wavfile
        
        # Convert to numpy and ensure correct format
        audio_np = audio_tensor.detach().cpu().numpy()
        
        # Handle stereo vs mono
        if audio_np.shape[0] == 2:  # Stereo
            audio_np = audio_np.T  # Transpose to [samples, channels]
        elif audio_np.shape[0] == 1:  # Mono
            audio_np = audio_np[0]  # Remove channel dimension
        
        # Normalize to 16-bit range
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_np = (audio_np * 32767).astype(np.int16)
        
        wavfile.write(filename, sample_rate, audio_np)
        return True
    except ImportError:
        print("⚠️  scipy not available, installing...")
        os.system("pip install scipy")
        return save_audio_as_wav(audio_tensor, filename, sample_rate)
    except Exception as e:
        print(f"❌ Error saving {filename}: {e}")
        return False


def build_model_from_config_dict(cfg: dict) -> DiffusionModel:
    """Build a DiffusionModel using keys saved in training checkpoint config dict.
    Falls back to reasonable defaults if some keys are missing.
    """
    # Some checkpoints (from train_diffusion.py) store `config.__dict__` under 'config'
    # We only need a subset to rebuild the model for inference.
    in_channels = cfg.get('in_channels', 2)
    channels = cfg.get('model_channels', [8, 32, 64, 128])
    factors = cfg.get('factors', [1, 4, 4, 2])
    items = cfg.get('items', [1, 2, 2, 2])
    attentions = cfg.get('attentions', [0, 0, 1, 1])
    attention_heads = cfg.get('attention_heads', 4)
    attention_features = cfg.get('attention_features', 32)

    model = DiffusionModel(
        net_t=UNetV0,
        in_channels=in_channels,
        channels=channels,
        factors=factors,
        items=items,
        attentions=attentions,
        attention_heads=attention_heads,
        attention_features=attention_features,
        diffusion_t=VDiffusion,
        sampler_t=VSampler,
    )
    return model


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[DiffusionModel, dict]:
    """Load a DiffusionModel and its saved state from a training checkpoint (.pth).
    Returns (model, meta) where meta contains useful fields like sample_rate.
    """
    # Handle checkpoints saved on Unix that pickle pathlib.PosixPath
    import pathlib
    if hasattr(pathlib, 'PosixPath') and hasattr(pathlib, 'WindowsPath'):
        try:
            pathlib.PosixPath = pathlib.WindowsPath  # type: ignore
        except Exception:
            pass

    # Explicitly set weights_only to False; we need the full dict (epoch, optimizer, etc.)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Try to reconstruct model architecture from checkpoint config
    cfg = ckpt.get('config') or {}

    try:
        model = build_model_from_config_dict(cfg)
    except Exception as e:
        print(f"⚠️  Couldn't rebuild model from checkpoint config ({e}). Using default small model.")
        model = DiffusionModel(
            net_t=UNetV0,
            in_channels=2,
            channels=[8, 32, 64, 128],
            factors=[1, 4, 4, 2],
            items=[1, 2, 2, 2],
            attentions=[0, 0, 1, 1],
            attention_heads=4,
            attention_features=32,
            diffusion_t=VDiffusion,
            sampler_t=VSampler,
        )

    model.to(device)

    # Load weights (be lenient to allow minor schema differences)
    state_dict = ckpt.get('model_state_dict') or ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"⚠️  Missing keys ({len(missing)}). Model may still work: {list(missing)[:8]}...")
    if unexpected:
        print(f"⚠️  Unexpected keys ({len(unexpected)}): {list(unexpected)[:8]}...")

    model.eval()

    # Derive metadata
    sample_rate = 16000
    if isinstance(cfg, dict):
        sr = cfg.get('sampling_rate') or cfg.get('sampling_rate'.upper())
        if sr:
            try:
                sample_rate = int(sr)
            except Exception:
                pass
    meta = {
        'sample_rate': sample_rate,
        'config': cfg,
        'epoch': ckpt.get('epoch'),
        'loss': ckpt.get('loss'),
    }
    return model, meta


def generate_quick_samples(steps_list=(5, 15, 30), sample_rate=16000, audio_length=2**15,
                           checkpoint_path: str | None = None, device: str = 'cpu'):
    """Generate a few quick audio samples. Optionally load a trained checkpoint."""
    print("🎵 Generating Audio Samples\n")
    
    # Create output directory
    output_dir = "generated_audio"
    os.makedirs(output_dir, exist_ok=True)
    
    torch_device = torch.device(device)
    overall_start = time.perf_counter()

    # Create or load model
    if checkpoint_path:
        print(f"📝 Loading model from checkpoint: {checkpoint_path}")
        model, meta = load_model_from_checkpoint(checkpoint_path, torch_device)
        # Prefer sample rate from checkpoint if present and not overridden
        if sample_rate is None:
            sample_rate = meta.get('sample_rate', 16000)
        print(f"✅ Checkpoint loaded (epoch={meta.get('epoch')}, loss={meta.get('loss')})")
    else:
        print("📝 Creating diffusion model (default small config)...")
        model = DiffusionModel(
            net_t=UNetV0,
            in_channels=2,  # Stereo
            channels=[8, 32, 64, 128],
            factors=[1, 4, 4, 2],
            items=[1, 2, 2, 2],
            attentions=[0, 0, 1, 1],
            attention_heads=4,
            attention_features=32,
            diffusion_t=VDiffusion,
            sampler_t=VSampler,
        ).to(torch_device)
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Ensure eval mode
    model.eval()
    
    # Generate samples with different step counts
    sample_configs = [(int(s), name) for s, name in zip(steps_list, ["quick", "medium", "high_quality"][:len(steps_list)])]
    per_sample_times = []
    
    for steps, quality_name in sample_configs:
        print(f"🎲 Generating {quality_name} sample ({steps} steps)...")
        lap_start = time.perf_counter()
        
        # Create noise
        noise = torch.randn(1, 2, audio_length, device=torch_device)
        
        # Generate audio
        t0 = time.perf_counter()
        with torch.no_grad():
            sample = model.sample(noise, num_steps=steps)
        infer_s = time.perf_counter() - t0
        per_sample_times.append(infer_s)
        
        # Normalize to prevent clipping
        sample = sample / (sample.abs().max() + 1e-8) * 0.8
        
        # Save audio
        filename = os.path.join(output_dir, f"diffusion_{quality_name}_{steps}steps.wav")
        full_path = os.path.abspath(filename)
        
        if save_audio_as_wav(sample[0].cpu(), filename, sample_rate or 16000):
            print(f"✅ Saved: {full_path}")
        else:
            print(f"❌ Failed to save: {full_path}")
        audio_sec = (audio_length / float(sample_rate or 16000))
        rtf = audio_sec / max(infer_s, 1e-8)
        lap_s = time.perf_counter() - lap_start
        print(f"   📊 Audio stats: shape={tuple(sample.shape)}, range=[{float(sample.min()):.3f}, {float(sample.max()):.3f}]")
        print(f"   ⏱️ Inference: {infer_s:.2f}s | audio_len={audio_sec:.2f}s | RTF={rtf:.2f}x")
        # print(f"   ⏲️ Mode time (lap): {lap_s:.2f}s\n")
    
    total_s = time.perf_counter() - overall_start
    if per_sample_times:
        avg_s = sum(per_sample_times) / len(per_sample_times)
        print(f"⏲️ Total generation time: {total_s:.2f}s | Avg per sample: {avg_s:.2f}s")
    else:
        print(f"⏲️ Total generation time: {total_s:.2f}s")
    
    return output_dir


if __name__ == "__main__":
    print("🎶 Simple Audio Diffusion Generator\n")
    
    parser = argparse.ArgumentParser(description="Generate audio samples with an Audio Diffusion model")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to a training checkpoint .pth (e.g., best_model.pth)")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use: cpu or cuda:0, etc.")
    parser.add_argument('--steps', type=int, nargs='*', default=[5, 15, 30], help="Sampling steps list for multiple outputs")
    parser.add_argument('--sample_rate', type=int, default=16000, help="Output WAV sample rate")
    parser.add_argument('--length', type=int, default=2**15, help="Audio length in samples for generation")
    args = parser.parse_args()

    try:
        output_dir = generate_quick_samples(
            steps_list=args.steps,
            sample_rate=args.sample_rate,
            audio_length=args.length,
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
        full_output_path = os.path.abspath(output_dir)
        
        print("🎉 Generation Complete!")
        print(f"📁 Audio files saved to: {full_output_path}")
        print(f"🎧 Open these WAV files in any audio player to listen!")
        
        # List the generated files
        if os.path.exists(output_dir):
            files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
            print(f"\n📋 Generated files ({len(files)}):")
            for file in sorted(files):
                full_file_path = os.path.join(full_output_path, file)
                print(f"   • {full_file_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc() 