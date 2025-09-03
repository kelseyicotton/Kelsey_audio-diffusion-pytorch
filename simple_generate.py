#!/usr/bin/env python3
"""
Simple audio generation script with robust WAV file saving
Works on Windows without additional audio backend dependencies.
"""

import torch
import numpy as np
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import os

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

def generate_quick_samples():
    """Generate a few quick audio samples."""
    print("🎵 Generating Audio Samples...\n")
    
    # Create output directory
    output_dir = "generated_audio"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple model
    print("📝 Creating diffusion model...")
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
    )
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Generate samples with different step counts
    sample_configs = [
        (5, "quick"),
        (15, "medium"), 
        (30, "high_quality")
    ]
    
    sample_rate = 16000
    audio_length = 2**15  # ~2 seconds at 16kHz
    
    for steps, quality_name in sample_configs:
        print(f"🎲 Generating {quality_name} sample ({steps} steps)...")
        
        # Create noise
        noise = torch.randn(1, 2, audio_length)
        
        # Generate audio
        with torch.no_grad():
            sample = model.sample(noise, num_steps=steps)
        
        # Normalize to prevent clipping
        sample = sample / sample.abs().max() * 0.8
        
        # Save audio
        filename = os.path.join(output_dir, f"diffusion_{quality_name}_{steps}steps.wav")
        full_path = os.path.abspath(filename)
        
        if save_audio_as_wav(sample[0], filename, sample_rate):
            print(f"✅ Saved: {full_path}")
        else:
            print(f"❌ Failed to save: {full_path}")
        
        print(f"   📊 Audio stats: shape={sample.shape}, range=[{sample.min():.3f}, {sample.max():.3f}]\n")
    
    return output_dir

if __name__ == "__main__":
    print("🎶 Simple Audio Diffusion Generator\n")
    
    try:
        output_dir = generate_quick_samples()
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