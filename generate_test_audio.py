#!/usr/bin/env python3
"""
Generate synthetic test audio files for training pipeline testing.
Creates 100 diverse audio files with different patterns and characteristics.
"""

import torch
import torchaudio
import numpy as np
import os
from pathlib import Path
import math

def generate_sine_wave(frequency, duration, sample_rate, amplitude=0.5):
    """Generate a sine wave."""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    wave = amplitude * torch.sin(2 * math.pi * frequency * t)
    return wave

def generate_chirp(start_freq, end_freq, duration, sample_rate, amplitude=0.5):
    """Generate a frequency sweep (chirp)."""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    # Linear chirp
    freq_t = start_freq + (end_freq - start_freq) * t / duration
    wave = amplitude * torch.sin(2 * math.pi * freq_t * t)
    return wave

def generate_noise_burst(duration, sample_rate, noise_type='white', amplitude=0.3):
    """Generate different types of noise."""
    samples = int(sample_rate * duration)
    
    if noise_type == 'white':
        wave = amplitude * torch.randn(samples)
    elif noise_type == 'pink':
        # Simple pink noise approximation
        white = torch.randn(samples)
        # Apply simple filter for pink-ish spectrum
        wave = amplitude * torch.cumsum(white, dim=0) / torch.sqrt(torch.arange(1, samples + 1).float())
        wave = wave - wave.mean()  # Remove DC
    elif noise_type == 'brown':
        # Brown noise (random walk)
        white = torch.randn(samples)
        wave = amplitude * torch.cumsum(white, dim=0)
        wave = wave / wave.std() * amplitude  # Normalize
    else:
        wave = amplitude * torch.randn(samples)
    
    return wave

def generate_harmonic_series(fundamental, num_harmonics, duration, sample_rate, amplitude=0.4):
    """Generate harmonic series (like musical instruments)."""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    wave = torch.zeros_like(t)
    
    for i in range(1, num_harmonics + 1):
        harmonic_amp = amplitude / i  # Decreasing amplitude
        wave += harmonic_amp * torch.sin(2 * math.pi * fundamental * i * t)
    
    return wave

def generate_am_modulated(carrier_freq, mod_freq, duration, sample_rate, amplitude=0.5):
    """Generate amplitude modulated signal."""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    carrier = torch.sin(2 * math.pi * carrier_freq * t)
    modulator = 0.5 * (1 + torch.sin(2 * math.pi * mod_freq * t))
    wave = amplitude * carrier * modulator
    return wave

def generate_fm_modulated(carrier_freq, mod_freq, mod_depth, duration, sample_rate, amplitude=0.5):
    """Generate frequency modulated signal."""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    modulator = mod_depth * torch.sin(2 * math.pi * mod_freq * t)
    instantaneous_freq = carrier_freq + modulator
    phase = 2 * math.pi * torch.cumsum(instantaneous_freq, dim=0) / sample_rate
    wave = amplitude * torch.sin(phase)
    return wave

def generate_pulse_train(frequency, duty_cycle, duration, sample_rate, amplitude=0.5):
    """Generate pulse train (square-ish waves)."""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    phase = (frequency * t) % 1.0
    wave = amplitude * (2 * (phase < duty_cycle).float() - 1)
    return wave

def add_envelope(wave, envelope_type='linear', attack=0.1, decay=0.1):
    """Add amplitude envelope to audio."""
    length = len(wave)
    envelope = torch.ones_like(wave)
    
    if envelope_type == 'linear':
        # Linear attack and decay
        attack_samples = int(attack * length)
        decay_samples = int(decay * length)
        
        # Attack
        envelope[:attack_samples] = torch.linspace(0, 1, attack_samples)
        # Decay
        envelope[-decay_samples:] = torch.linspace(1, 0, decay_samples)
    
    elif envelope_type == 'exponential':
        # Exponential decay
        decay_samples = int(decay * length)
        envelope[-decay_samples:] = torch.exp(-5 * torch.linspace(0, 1, decay_samples))
    
    return wave * envelope

def make_stereo(mono_wave, stereo_type='duplicate'):
    """Convert mono to stereo with different strategies."""
    if stereo_type == 'duplicate':
        return torch.stack([mono_wave, mono_wave])
    elif stereo_type == 'delay':
        # Add slight delay to right channel
        delay_samples = 100
        right = torch.cat([torch.zeros(delay_samples), mono_wave[:-delay_samples]])
        return torch.stack([mono_wave, right])
    elif stereo_type == 'phase':
        # Phase shift right channel
        return torch.stack([mono_wave, -mono_wave])
    elif stereo_type == 'filter':
        # Simple high-pass on right channel
        right = mono_wave - torch.cat([torch.zeros(1), mono_wave[:-1]])
        return torch.stack([mono_wave, right])
    else:
        return torch.stack([mono_wave, mono_wave])

def generate_test_audio_files(output_dir, num_files=100, sample_rate=16000):
    """Generate diverse test audio files."""
    
    # Create the base directory and audio subdirectory
    base_path = Path(output_dir)
    output_path = base_path / "audio"
    base_path.mkdir(exist_ok=True)
    output_path.mkdir(exist_ok=True)
    
    print(f"🎵 Generating {num_files} test audio files...")
    print(f"📁 Output directory: {output_path.absolute()}")
    
    # Define audio generation patterns
    patterns = [
        # Sine waves
        {'type': 'sine', 'freq_range': (50, 2000), 'duration_range': (2, 8)},
        # Chirps
        {'type': 'chirp', 'freq_range': (100, 1000), 'duration_range': (3, 6)},
        # Noise
        {'type': 'noise', 'noise_types': ['white', 'pink', 'brown'], 'duration_range': (2, 5)},
        # Harmonics
        {'type': 'harmonic', 'fundamental_range': (80, 400), 'duration_range': (3, 7)},
        # AM modulation
        {'type': 'am', 'carrier_range': (200, 800), 'mod_range': (5, 50), 'duration_range': (3, 6)},
        # FM modulation
        {'type': 'fm', 'carrier_range': (300, 1000), 'mod_range': (10, 100), 'duration_range': (3, 6)},
        # Pulse trains
        {'type': 'pulse', 'freq_range': (50, 500), 'duration_range': (2, 5)},
    ]
    
    for i in range(num_files):
        # Choose random pattern
        pattern = patterns[i % len(patterns)]
        
        # Generate base parameters
        duration = np.random.uniform(*pattern['duration_range'])
        
        if pattern['type'] == 'sine':
            frequency = np.random.uniform(*pattern['freq_range'])
            wave = generate_sine_wave(frequency, duration, sample_rate)
            filename = f"sine_{frequency:.0f}hz_{i:03d}.wav"
            
        elif pattern['type'] == 'chirp':
            start_freq = np.random.uniform(*pattern['freq_range'])
            end_freq = np.random.uniform(*pattern['freq_range'])
            wave = generate_chirp(start_freq, end_freq, duration, sample_rate)
            filename = f"chirp_{start_freq:.0f}-{end_freq:.0f}hz_{i:03d}.wav"
            
        elif pattern['type'] == 'noise':
            noise_type = np.random.choice(pattern['noise_types'])
            wave = generate_noise_burst(duration, sample_rate, noise_type)
            filename = f"noise_{noise_type}_{i:03d}.wav"
            
        elif pattern['type'] == 'harmonic':
            fundamental = np.random.uniform(*pattern['fundamental_range'])
            num_harmonics = np.random.randint(3, 8)
            wave = generate_harmonic_series(fundamental, num_harmonics, duration, sample_rate)
            filename = f"harmonic_{fundamental:.0f}hz_{num_harmonics}h_{i:03d}.wav"
            
        elif pattern['type'] == 'am':
            carrier = np.random.uniform(*pattern['carrier_range'])
            mod_freq = np.random.uniform(*pattern['mod_range'])
            wave = generate_am_modulated(carrier, mod_freq, duration, sample_rate)
            filename = f"am_{carrier:.0f}c_{mod_freq:.0f}m_{i:03d}.wav"
            
        elif pattern['type'] == 'fm':
            carrier = np.random.uniform(*pattern['carrier_range'])
            mod_freq = np.random.uniform(*pattern['mod_range'])
            mod_depth = np.random.uniform(10, 100)
            wave = generate_fm_modulated(carrier, mod_freq, mod_depth, duration, sample_rate)
            filename = f"fm_{carrier:.0f}c_{mod_freq:.0f}m_{i:03d}.wav"
            
        elif pattern['type'] == 'pulse':
            frequency = np.random.uniform(*pattern['freq_range'])
            duty_cycle = np.random.uniform(0.1, 0.9)
            wave = generate_pulse_train(frequency, duty_cycle, duration, sample_rate)
            filename = f"pulse_{frequency:.0f}hz_{duty_cycle:.1f}d_{i:03d}.wav"
        
        # Add random envelope
        envelope_type = np.random.choice(['linear', 'exponential'])
        attack = np.random.uniform(0.05, 0.2)
        decay = np.random.uniform(0.05, 0.3)
        wave = add_envelope(wave, envelope_type, attack, decay)
        
        # Convert to stereo with random strategy
        stereo_type = np.random.choice(['duplicate', 'delay', 'phase', 'filter'])
        stereo_wave = make_stereo(wave, stereo_type)
        
        # Normalize to prevent clipping
        stereo_wave = stereo_wave / stereo_wave.abs().max() * 0.8
        
        # Save file
        file_path = output_path / filename
        torchaudio.save(str(file_path), stereo_wave, sample_rate)
        
        if (i + 1) % 10 == 0:
            print(f"✅ Generated {i + 1}/{num_files} files...")
    
    print(f"🎉 Generated {num_files} test audio files!")
    return output_path

def create_local_test_config():
    """Create a local test config that points to our test audio."""
    config_content = """[audio]
sampling_rate = 16000
segment_length = 16384
channels = 2
normalize = True
dtype = float32

[dataset]
datapath = test_audio
test_dataset = test_audio
streaming = True
max_files = 100
shuffle = True
hop_size = 8192
num_workers = 0
check_audio = True
check_dataset = True

[model]
net_type = UNetV0
in_channels = 2
channels = [8, 32, 64]
factors = [1, 4, 4]
items = [1, 2, 2]
attentions = [0, 0, 1]
attention_heads = 4
attention_features = 32
diffusion_type = VDiffusion
sampler_type = VSampler
loss_function = mse_loss

[training]
epochs = 10
learning_rate = 0.0001
batch_size = 2
optimizer = AdamW
device = cuda:0
save_every = 2
sample_every = 1
max_batches_per_epoch = 10
gradient_clip = 1.0

[paths]
workspace = D:\\kelse\\03_Repositories\\Kelsey_audio-diffusion-pytorch
checkpoint_dir = checkpoints_local_test
sample_dir = samples_local_test
log_dir = logs_local_test

[monitoring]
generate_samples = True
num_samples_per_epoch = 2
sample_steps = 5
save_best_model = True
track_loss = True
progress_bar = True

[notes]
additional_notes = Local test with synthetic audio files
Calculations:
segment_length = 16384 samples = ~1.02 seconds at 16kHz
100 synthetic audio files with diverse patterns
Very small model for fast local testing

[extra]
description = 20250113_audio-diffusion-kelsey-local-test-synthetic
start_time = 
end_time = 
time_elapsed = 
best_loss = 
total_samples_generated = 
gpu_memory_used = 
notes_during_training = 

[resume]
resume_from_checkpoint = False
checkpoint_path = 
start_epoch = 0
"""
    
    with open('config_local_test.ini', 'w') as f:
        f.write(config_content)
    
    print("✅ Created config_local_test.ini")

if __name__ == "__main__":
    print("🎵 Synthetic Audio Generator for Training Pipeline Test")
    
    # Generate test audio files
    output_dir = generate_test_audio_files("training_audio", num_files=100)
    
    # Create matching config file
    create_local_test_config()
    
    print(f"\n🚀 Local Test Setup Complete!")
    print(f"📁 Audio files: {output_dir.absolute()}")
    print(f"📄 Config file: config_local_test.ini")
    print(f"\n🎯 To start training:")
    print(f"   python train_diffusion.py --config config_local_test.ini")
    
    # Show some stats
    audio_files = list(output_dir.glob("*.wav"))
    print(f"\n📊 Generated Files Summary:")
    print(f"   Total files: {len(audio_files)}")
    
    # Count by type
    types = {}
    for f in audio_files:
        file_type = f.name.split('_')[0]
        types[file_type] = types.get(file_type, 0) + 1
    
    for audio_type, count in types.items():
        print(f"   {audio_type}: {count} files")
    
    print(f"\n🎧 Sample filenames:")
    for f in sorted(audio_files)[:5]:
        print(f"   {f.name}")
    print(f"   ... and {len(audio_files) - 5} more") 