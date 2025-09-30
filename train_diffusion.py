#!/usr/bin/env python3
"""
Training script for Audio Diffusion Model
Uses the adapted dataloader to train on real audio data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
from pathlib import Path
import json
from tqdm import tqdm
import configparser
import ast
import argparse
import logging
import sys
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", message=".*TorchCodec.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*", category=UserWarning)
warnings.filterwarnings("ignore", module="torchaudio")

from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
from audio_diffusion_pytorch.dataset import create_diffusion_dataloader
from audio_diffusion_pytorch.utils import inject_lora  # LoRA injection
import torchaudio

class TeeOutput:
    """Custom class to write output to both console and file."""
    def __init__(self, file_path, original_stream):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.original_stream = original_stream
        
    def write(self, text):
        self.original_stream.write(text)
        self.file.write(text)
        self.file.flush()  # Ensure immediate write
        
    def flush(self):
        self.original_stream.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()

def setup_logging(log_dir):
    """Setup logging to capture all printed output."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_output_{timestamp}.log"
    
    # Setup tee output for stdout
    tee_stdout = TeeOutput(log_file, sys.stdout)
    sys.stdout = tee_stdout
    
    print(f"📝 All output will be logged to: {log_file}")
    return tee_stdout

class DiffusionConfig:
    """Configuration class that reads from config.ini file."""
    
    def __init__(self, config_path="config.ini"):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        
        # Check if config file exists
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Read config file
        self.config.read(config_path)
        
        # Parse all sections (extra config first for description)
        self._parse_audio_config()
        self._parse_dataset_config()
        self._parse_model_config()
        self._parse_training_config()
        self._parse_extra_config()
        self._parse_paths_config()
        self._parse_monitoring_config()
        self._parse_resume_config()
        
        print(f"✅ Loaded configuration from: {config_path}")
    
    def _parse_audio_config(self):
        """Parse [audio] section."""
        audio = self.config['audio']
        self.sampling_rate = audio.getint('sampling_rate')
        self.segment_length = audio.getint('segment_length')
        self.channels = audio.getint('channels')
        self.normalize = audio.getboolean('normalize')
        self.dtype = getattr(torch, audio.get('dtype'))
    
    def _parse_dataset_config(self):
        """Parse [dataset] section."""
        dataset = self.config['dataset']
        self.datapath = dataset.get('datapath')
        self.streaming = dataset.getboolean('streaming')
        self.max_files = dataset.get('max_files') or None
        if self.max_files:
            self.max_files = int(self.max_files)
        self.shuffle = dataset.getboolean('shuffle')
        self.hop_size = dataset.getint('hop_size')
        self.num_workers = dataset.getint('num_workers')
    
    def _parse_model_config(self):
        """Parse [model] section."""
        model = self.config['model']
        
        # Network type
        net_type = model.get('net_type')
        if net_type == 'UNetV0':
            self.net_t = UNetV0
        else:
            raise ValueError(f"Unknown net_type: {net_type}")
        
        # Model architecture
        self.in_channels = model.getint('in_channels')
        self.model_channels = ast.literal_eval(model.get('channels'))
        self.factors = ast.literal_eval(model.get('factors'))
        self.items = ast.literal_eval(model.get('items'))
        self.attentions = ast.literal_eval(model.get('attentions'))
        self.attention_heads = model.getint('attention_heads')
        self.attention_features = model.getint('attention_features')
        
        # Diffusion components
        diffusion_type = model.get('diffusion_type')
        if diffusion_type == 'VDiffusion':
            self.diffusion_t = VDiffusion
        else:
            raise ValueError(f"Unknown diffusion_type: {diffusion_type}")
        
        sampler_type = model.get('sampler_type')
        if sampler_type == 'VSampler':
            self.sampler_t = VSampler
        else:
            raise ValueError(f"Unknown sampler_type: {sampler_type}")
        
        # Loss function
        loss_function = model.get('loss_function')
        if loss_function == 'mse_loss':
            self.loss_fn = torch.nn.functional.mse_loss
        else:
            raise ValueError(f"Unknown loss_function: {loss_function}")
    
    def _parse_training_config(self):
        """Parse [training] section."""
        training = self.config['training']
        self.epochs = training.getint('epochs')
        self.learning_rate = training.getfloat('learning_rate')
        self.batch_size = training.getint('batch_size')
        self.optimizer_name = training.get('optimizer')
        self.device = torch.device(training.get('device'))
        self.save_every = training.getint('save_every')
        self.sample_every = training.getint('sample_every')
        self.max_batches_per_epoch = training.getint('max_batches_per_epoch')
        self.gradient_clip = training.getfloat('gradient_clip')
    
    def _parse_paths_config(self):
        """Parse [paths] section."""
        paths = self.config['paths']
        self.workspace = Path(paths.get('workspace'))
        
        # Auto-generate checkpoint directory name if it contains placeholders
        checkpoint_dir_template = paths.get('checkpoint_dir')
        if '{timestamp}' in checkpoint_dir_template or '{experiment}' in checkpoint_dir_template:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = self.description or 'experiment'
            checkpoint_dir_name = checkpoint_dir_template.format(
                timestamp=timestamp,
                experiment=experiment_name
            )
        else:
            checkpoint_dir_name = checkpoint_dir_template
            
        self.checkpoint_dir = self.workspace / checkpoint_dir_name
        # Organize all outputs under checkpoint_dir for cleanliness
        self.checkpoints_subdir = self.checkpoint_dir / 'checkpoints'
        self.sample_dir = self.checkpoint_dir / 'samples'
        self.log_dir = self.checkpoint_dir / 'logs'
    
    def _parse_monitoring_config(self):
        """Parse [monitoring] section."""
        monitoring = self.config['monitoring']
        self.generate_samples = monitoring.getboolean('generate_samples')
        self.num_samples_per_epoch = monitoring.getint('num_samples_per_epoch')
        self.sample_steps = monitoring.getint('sample_steps')
        self.save_best_model = monitoring.getboolean('save_best_model')
    
    def _parse_extra_config(self):
        """Parse [extra] section."""
        extra = self.config['extra']
        self.description = extra.get('description')
    
    def _parse_resume_config(self):
        """Parse [resume] section."""
        resume = self.config['resume']
        self.resume_from_checkpoint = resume.getboolean('resume_from_checkpoint') if resume.get('resume_from_checkpoint') else False
        self.checkpoint_path = resume.get('checkpoint_path')
        self.start_epoch = resume.getint('start_epoch')
    
    def get_model_config(self):
        """Get model configuration dictionary for DiffusionModel."""
        return {
            'net_t': self.net_t,
            'in_channels': self.in_channels,
            'channels': self.model_channels,
            'factors': self.factors,
            'items': self.items,
            'attentions': self.attentions,
            'attention_heads': self.attention_heads,
            'attention_features': self.attention_features,
            'diffusion_t': self.diffusion_t,
            'sampler_t': self.sampler_t,
            'loss_fn': self.loss_fn,
        }
    
    def get_dataloader_config(self):
        """Get dataloader configuration dictionary."""
        return {
            'audio_folder': self.datapath,
            'batch_size': self.batch_size,
            'sampling_rate': self.sampling_rate,
            'segment_length': self.segment_length,
            'channels': self.channels,
            'streaming': self.streaming,
            'max_files': self.max_files,
            'num_workers': self.num_workers,
        }
    
    def create_directories(self):
        """Create all necessary directories."""
        directories = [self.checkpoint_dir, self.checkpoints_subdir, self.sample_dir, self.log_dir]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {directory}")
    
    def print_summary(self):
        """Print a summary of the current configuration."""
        print("\n🔧 Configuration Summary:")
        print(f"📊 Model: {self.net_t.__name__} with {len(self.model_channels)} layers")
        print(f"🎵 Audio: {self.sampling_rate}Hz, {self.segment_length} samples (~{self.segment_length/self.sampling_rate:.1f}s)")
        print(f"📁 Data: {self.datapath}")
        print(f"🏋️  Training: {self.epochs} epochs, batch_size={self.batch_size}, lr={self.learning_rate}")
        print(f"🖥️  Device: {self.device}")
        print(f"💾 Checkpoints: Every {self.save_every} epochs")
        print(f"🎧 Samples: Every {self.sample_every} epochs")

class DiffusionTrainer:
    """Main training class for diffusion models."""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.device = config.device
        
        # Initialize tensorboard logging
        self.writer = SummaryWriter(log_dir=str(config.log_dir))
        print(f"📊 Tensorboard logging to: {config.log_dir}")
        
        # Initialize model
        print("🎵 Initializing diffusion model...")
        self.model = DiffusionModel(**config.get_model_config()).to(self.device)
        
        # Inject LoRA adapters into selected Conv1d layers for PEFT
        lora_params = inject_lora(self.model, rank=4, alpha=8)
        # Freeze all base params; LoRA params remain trainable
        for p in self.model.parameters():
            p.requires_grad = False
        for p in lora_params:
            p.requires_grad = True
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Log model architecture to tensorboard
        self.writer.add_text('Model/Architecture', f"""
        **Model Type**: {config.net_t.__name__}
        **Parameters**: {total_params:,} total, {trainable_params:,} trainable
        **Channels**: {config.model_channels}
        **Factors**: {config.factors}
        **Items**: {config.items}
        **Attentions**: {config.attentions}
        """)
        
        # Initialize optimizer
        if config.optimizer_name == 'AdamW':
            params = lora_params if len(lora_params) > 0 else list(self.model.parameters())
            self.optimizer = optim.AdamW(params, lr=config.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer_name}")
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.global_step = 0
    
    def __del__(self):
        """Cleanup tensorboard writer when trainer is destroyed."""
        if hasattr(self, 'writer'):
            self.writer.close()
    
    def save_config_copy(self):
        """Save a copy of the config file to the checkpoint directory for reproducibility."""
        import shutil
        import configparser
        
        # Create a modified copy of the config with experiment info
        config_backup_path = self.config.checkpoint_dir / f"config_used.ini"
        
        # Read the original config
        backup_config = configparser.ConfigParser()
        backup_config.read(self.config.config_path)
        
        # Add experiment information to [extra] section
        if not backup_config.has_section('extra'):
            backup_config.add_section('extra')
        
        # Update the extra section with experiment details
        backup_config.set('extra', 'experiment_name', self.config.description or 'default')
        backup_config.set('extra', 'training_completed', datetime.now().isoformat())
        backup_config.set('extra', 'final_loss', str(self.best_loss))
        backup_config.set('extra', 'total_steps', str(self.global_step))
        backup_config.set('extra', 'checkpoint_directory', str(self.config.checkpoint_dir))
        
        # Save the modified config
        with open(config_backup_path, 'w') as f:
            backup_config.write(f)
        print(f"📋 Config file (with experiment info) saved to: {config_backup_path}")
        
        # Also save a JSON version with final training results
        config_summary = {
            'training_completed': datetime.now().isoformat(),
            'experiment_name': self.config.description or 'default',
            'final_loss': float(self.best_loss),
            'total_epochs': int(self.config.epochs),
            'total_steps': int(self.global_step),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'config_file_used': str(self.config.config_path),
            'checkpoint_directory': str(self.config.checkpoint_dir)
        }
        
        summary_path = self.config.checkpoint_dir / "training_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(config_summary, f, indent=2)
        print(f"📊 Training summary saved to: {summary_path}")
        
    def create_dataloader(self):
        """Create training dataloader."""
        print(f"📁 Setting up dataloader for: {self.config.datapath}")
        
        dataloader = create_diffusion_dataloader(**self.config.get_dataloader_config())
        
        return dataloader
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.checkpoints_subdir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.checkpoints_subdir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 New best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['loss']
        print(f"✅ Loaded checkpoint from epoch {self.current_epoch}")
    
    def generate_samples(self, epoch, num_samples=3):
        """Generate audio samples during training for monitoring."""
        print(f"🎲 Generating samples at epoch {epoch}...")
        
        self.model.eval()
        with torch.no_grad():
            for i in range(self.config.num_samples_per_epoch):
                # Generate sample
                noise = torch.randn(1, self.config.channels, self.config.segment_length).to(self.device)
                sample = self.model.sample(noise, num_steps=self.config.sample_steps)
                
                # Better audio post-processing to reduce crackling
                # 1. Clamp to reasonable range first
                sample = torch.clamp(sample, -1.0, 1.0)
                
                # 2. Soft normalization to prevent harsh clipping
                max_val = sample.abs().max()
                if max_val > 0.1:  # Only normalize if signal is significant
                    sample = sample / max_val * 0.7  # More conservative scaling
                
                # 3. Apply gentle tanh saturation to smooth harsh edges
                sample = torch.tanh(sample * 0.9) * 0.8
                sample_path = os.path.join(
                    self.config.sample_dir, 
                    f'epoch_{epoch:03d}_sample_{i+1}.wav'
                )
                
                try:
                    torchaudio.save(sample_path, sample[0].cpu(), self.config.sampling_rate)
                    
                    # Log audio to tensorboard (first sample only to avoid clutter)
                    if i == 0:
                        self.writer.add_audio(
                            f'Generated_Audio/Epoch_{epoch}', 
                            sample[0].cpu(), 
                            epoch, 
                            sample_rate=self.config.sampling_rate
                        )
                except:
                    # Fallback: save as numpy
                    import numpy as np
                    np.save(sample_path.replace('.wav', '.npy'), sample[0].cpu().numpy())
        
        self.model.train()
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Use tqdm for progress bar
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, audio_batch in enumerate(pbar):
            # Move to device
            audio_batch = audio_batch.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass (compute diffusion loss)
            loss = self.model(audio_batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if configured
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log to tensorboard every 10 steps
            if self.global_step % 10 == 0:
                self.writer.add_scalar('Loss/Training_Step', loss.item(), self.global_step)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Break after max batches per epoch (configurable)
            if batch_idx >= self.config.max_batches_per_epoch:
                break
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def train(self):
        """Main training loop."""
        print(f"🚀 Starting training on {self.device}")
        print(f"📊 Training for {self.config.epochs} epochs")
        
        # Create dataloader
        dataloader = self.create_dataloader()
        
        for epoch in range(self.current_epoch, self.config.epochs):
            start_time = time.time()
            
            # Train one epoch
            avg_loss = self.train_epoch(dataloader, epoch)
            
            epoch_time = time.time() - start_time
            print(f"📈 Epoch {epoch}: loss={avg_loss:.4f}, time={epoch_time:.1f}s")
            
            # Log epoch metrics to tensorboard
            self.writer.add_scalar('Loss/Epoch', avg_loss, epoch)
            self.writer.add_scalar('Time/Epoch_Duration', epoch_time, epoch)
            self.writer.add_scalar('Training/Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss
                self.writer.add_scalar('Loss/Best', self.best_loss, epoch)
            
            if epoch % self.config.save_every == 0 or is_best:
                self.save_checkpoint(epoch, avg_loss, is_best)
            
            # Generate samples
            if self.config.generate_samples and epoch % self.config.sample_every == 0:
                self.generate_samples(epoch)
        
        print("🎉 Training completed!")
        
        # Log final training summary
        self.writer.add_text('Training/Summary', f"""
        **Training Completed Successfully!**
        - **Total Epochs**: {self.config.epochs}
        - **Final Loss**: {self.best_loss:.4f}
        - **Total Steps**: {self.global_step}
        - **Model Parameters**: {sum(p.numel() for p in self.model.parameters()):,}
        """)
        
        # Save copy of config file for reproducibility
        self.save_config_copy()
        
        # Close tensorboard writer
        self.writer.close()
        print(f"📊 Tensorboard logs saved to: {self.config.log_dir}")
        print(f"⏰ Training session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main training function using config file."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Audio Diffusion Model')
    parser.add_argument('--config', '-c', 
                       type=str, 
                       default='config.ini',
                       help='Path to config file (default: config.ini)')
    parser.add_argument('--resume', '-r',
                       type=str,
                       default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--experiment', '-e',
                       type=str,
                       default=None,
                       help='Override experiment name for directory naming')
    
    args = parser.parse_args()
    
    # Load config from specified file
    try:
        config = DiffusionConfig(args.config)
        # Override experiment name if provided
        if args.experiment:
            config.description = args.experiment
    except FileNotFoundError:
        print(f"❌ Config file not found: {args.config}")
        print(f"💡 Please make sure the config file exists")
        print(f"📝 Example: python train_diffusion.py --config my_experiment.ini")
        return
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return
    
    # Setup logging to capture all output
    tee_output = setup_logging(config.log_dir)
    
    # Log training session start
    print(f"🚀 Training session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📄 Config file: {args.config}")
    print(f"🖥️  Working directory: {os.getcwd()}")
    
    # Override resume settings if provided via command line
    if args.resume:
        config.resume_from_checkpoint = True
        config.checkpoint_path = args.resume
        print(f"🔄 Will resume training from: {args.resume}")
    
    # Create directories
    config.create_directories()
    
    # Print configuration summary
    print(f"🎵 Audio Diffusion Training")
    print(f"📄 Config file: {args.config}")
    config.print_summary()
    
    # Check if audio folder exists
    if not os.path.exists(config.datapath):
        print(f"❌ Audio folder not found: {config.datapath}")
        print(f"💡 Please update the datapath in {args.config} or create the folder")
        return
    
    # Create trainer and start training
    trainer = DiffusionTrainer(config)
    
    # Resume from checkpoint if specified
    if config.resume_from_checkpoint and config.checkpoint_path:
        trainer.load_checkpoint(config.checkpoint_path)
    
    try:
        trainer.train()
    finally:
        # Ensure output logging is properly closed
        if 'tee_output' in locals():
            tee_output.close()
            sys.stdout = tee_output.original_stream

if __name__ == "__main__":
    main() 