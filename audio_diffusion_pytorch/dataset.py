#!/usr/bin/env python3
"""
Audio Dataset adapted for Diffusion Model Training
Based on user's existing IterableAudioDataset but modified for diffusion requirements.
"""

import torch
import torchaudio
import librosa
from torch.utils.data import IterableDataset, Dataset
import pathlib
from pathlib import Path
import numpy as np
from itertools import chain, cycle
import random

class DiffusionAudioDataset(IterableDataset):
    """
    Adapted from user's IterableAudioDataset for diffusion model training.
    Key changes:
    - Returns audio in [channels, length] format (diffusion expects this)
    - Supports both mono and stereo
    - Configurable segment length for diffusion training
    """

    def __init__(self, 
                 audio_folder, 
                 sampling_rate=16000,
                 segment_length=2**15,  # ~2 seconds at 16kHz
                 hop_size=None,
                 channels=2,  # 1 for mono, 2 for stereo
                 dtype=torch.float32, 
                 device=torch.device('cpu'), 
                 shuffle=True,
                 normalize=True,
                 use_windowing=True):  # Enable Hann windowing by default
        
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.hop_size = hop_size or segment_length // 2  # Default: 50% overlap
        self.channels = channels
        self.dtype = dtype
        self.device = device
        self.shuffle = shuffle
        self.normalize = normalize
        self.use_windowing = use_windowing

        if isinstance(audio_folder, pathlib.PurePath):
            self.audio_folder = audio_folder
        else: 
            self.audio_folder = Path(audio_folder)

        # Support multiple audio formats
        audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
        self.audio_file_list = []
        for ext in audio_extensions:
            self.audio_file_list.extend(list(self.audio_folder.glob(ext)))
        
        self.num_files = len(self.audio_file_list)
        print(f"📁 Found {self.num_files} audio files in {self.audio_folder}")

    @property
    def shuffled_data_list(self):
        """Shuffle the audio file list for variety."""
        return random.sample(self.audio_file_list, len(self.audio_file_list))

    def process_data(self, audio_file):
        """Process a single audio file into training segments."""
        try:
            # Load audio with torchaudio
            audio_tensor, audio_sr = torchaudio.load(audio_file)
            
            # Resample if needed
            if audio_sr != self.sampling_rate:
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor, audio_sr, self.sampling_rate
                )
            
            # Handle channel configuration
            if self.channels == 1:  # Mono
                if audio_tensor.shape[0] > 1:
                    # Convert stereo to mono (average channels)
                    audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
            elif self.channels == 2:  # Stereo
                if audio_tensor.shape[0] == 1:
                    # Convert mono to stereo (duplicate channel)
                    audio_tensor = audio_tensor.repeat(2, 1)
                elif audio_tensor.shape[0] > 2:
                    # Take first 2 channels
                    audio_tensor = audio_tensor[:2, :]
            
            # Normalize audio if requested
            if self.normalize:
                # Normalize to [-1, 1] range
                max_val = audio_tensor.abs().max()
                if max_val > 0:
                    audio_tensor = audio_tensor / max_val
            
            # Convert to numpy for segmentation
            audio_np = audio_tensor.numpy()
            audio_length = audio_np.shape[1]
            
            # Skip files that are too short
            if audio_length < self.segment_length:
                return
            
            # Generate overlapping segments
            num_segments = (audio_length - self.segment_length) // self.hop_size + 1
            
            for i in range(num_segments):
                seg_start = i * self.hop_size
                seg_end = seg_start + self.segment_length
                segment = audio_np[:, seg_start:seg_end]  # [channels, length]
                
                # Convert back to tensor
                segment_tensor = torch.tensor(segment, dtype=self.dtype)
                
                # Apply Hann windowing to smooth segment boundaries and reduce artifacts
                if self.use_windowing:
                    hann_window = torch.hann_window(self.segment_length, dtype=self.dtype)
                    # Apply window to each channel
                    for ch in range(segment_tensor.shape[0]):
                        segment_tensor[ch] = segment_tensor[ch] * hann_window
                
                # Move to device if needed
                if self.device.type == "cuda":
                    segment_tensor = segment_tensor.to(self.device)
                
                yield segment_tensor
                
        except Exception as e:
            print(f"⚠️  Error processing {audio_file}: {e}")
            return

    def get_stream(self, audio_file_list):
        """Create infinite stream of audio segments."""
        return chain.from_iterable(map(self.process_data, cycle(audio_file_list)))
        
    def __iter__(self):
        if self.shuffle:
            return self.get_stream(self.shuffled_data_list)
        else: 
            return self.get_stream(self.audio_file_list)

class DiffusionAudioBatchDataset(Dataset):
    """
    Non-streaming version for smaller datasets that fit in memory.
    Useful for testing and small-scale experiments.
    """
    
    def __init__(self, 
                 audio_folder,
                 sampling_rate=16000,
                 segment_length=2**15,
                 channels=2,
                 normalize=True,
                 max_files=None,
                 use_windowing=True):
        
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.channels = channels
        self.normalize = normalize
        self.use_windowing = use_windowing
        
        # Load all audio files into memory
        self.segments = []
        
        audio_folder = Path(audio_folder)
        audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(list(audio_folder.glob(ext)))
        
        if max_files:
            audio_files = audio_files[:max_files]
        
        print(f"📁 Loading {len(audio_files)} audio files into memory...")
        
        for audio_file in audio_files:
            try:
                audio_tensor, audio_sr = torchaudio.load(audio_file)
                
                # Resample if needed
                if audio_sr != self.sampling_rate:
                    audio_tensor = torchaudio.functional.resample(
                        audio_tensor, audio_sr, self.sampling_rate
                    )
                
                # Handle channels
                if self.channels == 1 and audio_tensor.shape[0] > 1:
                    audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
                elif self.channels == 2:
                    if audio_tensor.shape[0] == 1:
                        audio_tensor = audio_tensor.repeat(2, 1)
                    elif audio_tensor.shape[0] > 2:
                        audio_tensor = audio_tensor[:2, :]
                
                # Normalize
                if self.normalize:
                    max_val = audio_tensor.abs().max()
                    if max_val > 0:
                        audio_tensor = audio_tensor / max_val
                
                # Split into segments
                audio_length = audio_tensor.shape[1]
                if audio_length >= self.segment_length:
                    num_segments = audio_length // self.segment_length
                    # Create Hann window for this segment length
                    hann_window = torch.hann_window(self.segment_length, dtype=audio_tensor.dtype)
                    
                    for i in range(num_segments):
                        start = i * self.segment_length
                        end = start + self.segment_length
                        segment = audio_tensor[:, start:end]
                        
                        # Apply Hann windowing to smooth boundaries
                        if self.use_windowing:
                            for ch in range(segment.shape[0]):
                                segment[ch] = segment[ch] * hann_window
                        
                        self.segments.append(segment)
                        
            except Exception as e:
                print(f"⚠️  Error loading {audio_file}: {e}")
                continue
        
        print(f"✅ Loaded {len(self.segments)} audio segments")
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        return self.segments[idx]

def create_diffusion_dataloader(audio_folder, 
                               batch_size=4,
                               sampling_rate=16000,
                               segment_length=2**15,
                               channels=2,
                               streaming=True,
                               max_files=None,
                               num_workers=0,
                               use_windowing=True):
    """
    Create a dataloader suitable for diffusion model training.
    
    Args:
        audio_folder: Path to folder containing audio files
        batch_size: Number of audio segments per batch
        sampling_rate: Target sample rate (16kHz recommended for fast training)
        segment_length: Length of audio segments (2**15 = ~2 seconds at 16kHz)
        channels: 1 for mono, 2 for stereo
        streaming: If True, use streaming dataset (for large datasets)
        max_files: Limit number of files (useful for testing)
        num_workers: Number of worker processes
    
    Returns:
        DataLoader ready for training
    """
    
    if streaming:
        dataset = DiffusionAudioDataset(
            audio_folder=audio_folder,
            sampling_rate=sampling_rate,
            segment_length=segment_length,
            hop_size=segment_length // 2,  # 50% overlap
            channels=channels,
            device=torch.device('cpu'),  # Load on CPU, move to GPU in training loop
            shuffle=True,
            use_windowing=use_windowing
        )
        # For IterableDataset, don't shuffle in DataLoader
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False  # Shuffling handled in dataset
        )
    else:
        dataset = DiffusionAudioBatchDataset(
            audio_folder=audio_folder,
            sampling_rate=sampling_rate,
            segment_length=segment_length,
            channels=channels,
            max_files=max_files,
            use_windowing=use_windowing
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
    
    return dataloader

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Diffusion Audio Dataset")
    
    # Test with a small dataset
    test_folder = Path("test_audio")  # You'll need to create this with some audio files
    
    if test_folder.exists():
        print(f"Testing with audio folder: {test_folder}")
        
        # Test streaming dataset
        dataloader = create_diffusion_dataloader(
            audio_folder=test_folder,
            batch_size=2,
            segment_length=2**14,  # 1 second for fast testing
            channels=2,
            streaming=True,
            max_files=5
        )
        
        print("📊 Testing dataloader...")
        for i, batch in enumerate(dataloader):
            print(f"Batch {i}: shape={batch.shape}, range=[{batch.min():.3f}, {batch.max():.3f}]")
            if i >= 2:  # Test first 3 batches
                break
        
        print("✅ Dataset test complete!")
    else:
        print(f"⚠️  Test folder {test_folder} not found. Create it with some audio files to test.") 