#!/usr/bin/env python3
"""
SmolLM2-135M training on AG News with DDP and performance optimizations
- DistributedDataParallel (DDP)
- Mixed Precision with bfloat16
- TF32 and cudnn optimizations
- Gradient accumulation
- SafeTensors for secure model loading
"""

import os
import argparse
import time
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType


# ============================================================================
# DDP Setup and Utilities
# ============================================================================
def setup_ddp(rank, world_size):
    """Initialize DDP"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


def is_main_process():
    """Check if current process is main"""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    """Get current rank"""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    """Get world size"""
    return dist.get_world_size() if dist.is_initialized() else 1


# ============================================================================
# AG News Dataset Wrapper
# ============================================================================
class AGNewsDataset(Dataset):
    """
    AG News dataset wrapper for PyTorch
    """
    def __init__(self, split, tokenizer, max_length=512):
        """
        Args:
            split: 'train' or 'test'
            tokenizer: SmolLM2 tokenizer
            max_length: maximum sequence length
        """
        self.dataset = load_dataset('ag_news', split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        label = item['label']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# Using SmolLM2-135M from Transformers
# ============================================================================
# We use AutoModelForSequenceClassification with SmolLM2-135M


# ============================================================================
# Evaluation Function
# ============================================================================
def evaluate_accuracy(model, loader, device, dtype=torch.bfloat16):
    """Evaluate model accuracy"""
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        # Use torch.inference_mode for better performance
        with torch.inference_mode():
            for batch in loader:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                
                with autocast('cuda', dtype=dtype):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    predictions = outputs.logits.argmax(dim=-1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
    
    model.train()  # Set back to training mode
    return correct / total


# ============================================================================
# EMA (Exponential Moving Average)
# ============================================================================
class EMA:
    """Exponential Moving Average of model parameters"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        """Return state dict for saving"""
        return {
            'decay': self.decay,
            'shadow': self.shadow
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint"""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']


# ============================================================================
# Plotting Function
# ============================================================================
def plot_training_curves(train_losses, test_accuracies, learning_rates, epoch_times, save_path):
    """Plot and save training curves"""
    if not is_main_process():
        return
    
    epochs_range = np.arange(1, len(train_losses) + 1)
    fig = plt.figure(figsize=(18, 10))
    
    # Training Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, train_losses, marker='o', markersize=3, label="Train Loss", color='blue')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss Curve", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Test Accuracy
    plt.subplot(2, 3, 2)
    plt.plot(epochs_range, test_accuracies, marker='s', markersize=3, color='green', label="Test Accuracy")
    best_acc = max(test_accuracies)
    best_epoch = test_accuracies.index(best_acc) + 1
    plt.axhline(y=best_acc, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_acc:.4f} (Epoch {best_epoch})')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Test Accuracy Curve", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Learning Rate
    plt.subplot(2, 3, 3)
    plt.plot(epochs_range, learning_rates, marker='^', markersize=3, color='orange', label="Learning Rate")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.title("Learning Rate Schedule", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Epoch Time
    plt.subplot(2, 3, 4)
    plt.plot(epochs_range, epoch_times, marker='o', markersize=2, color='purple', label="Epoch Time")
    plt.axhline(y=np.mean(epoch_times), color='r', linestyle='--', label=f'Mean: {np.mean(epoch_times):.2f}s')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.title("Training Time per Epoch", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Training summary
    plt.subplot(2, 3, 5)
    info_text = f"Training Summary\n\n"
    info_text += f"Dataset: AG News\n"
    info_text += f"Model: SmolLM2-135M\n\n"
    info_text += f"Results:\n"
    info_text += f"  Epochs: {len(train_losses)}\n"
    info_text += f"  Final Loss: {train_losses[-1]:.6f}\n"
    info_text += f"  Best Test Acc: {max(test_accuracies):.4f}\n"
    info_text += f"  Final Test Acc: {test_accuracies[-1]:.4f}\n\n"
    info_text += f"Training Time:\n"
    info_text += f"  Avg/Epoch: {np.mean(epoch_times):.2f}s\n"
    info_text += f"  Total Time: {sum(epoch_times)/3600:.2f}h\n"
    
    plt.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center', 
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Training Function
# ============================================================================
def train_worker(rank, world_size, args):
    """Training worker for each GPU"""
    # Setup DDP
    setup_ddp(rank, world_size)
    
    # Set random seed
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    
    # Performance optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device = torch.device(f"cuda:{rank}")
    
    if is_main_process():
        print(f"\n{'='*80}")
        print(f"SmolLM2-135M Training on AG News with DDP")
        print(f"{'='*80}")
        print(f"Using {world_size} GPU(s) with DDP")
        for i in range(world_size):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"\nPerformance Optimizations:")
        print(f"  - DistributedDataParallel (DDP)")
        print(f"  - Mixed Precision: bfloat16")
        print(f"  - TF32: Enabled")
        print(f"  - cudnn.benchmark: True")
    
    # Load tokenizer from Hugging Face Hub with authentication token
    if is_main_process():
        print(f"\nLoading SmolLM2-135M tokenizer from Hugging Face Hub...")
    
    # HF token for authentication
    hf_token = "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"
    
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        token=hf_token
    )
    
    # Set pad token if not already set (critical for batch processing)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if is_main_process():
        print(f"  ✓ Loaded tokenizer from HuggingFaceTB/SmolLM2-135M")
        print(f"  ✓ Pad token set to: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    # Load datasets
    if is_main_process():
        print(f"Loading AG News dataset...")
    
    train_dataset = AGNewsDataset('train', tokenizer, max_length=args.max_length)
    test_dataset = AGNewsDataset('test', tokenizer, max_length=args.max_length)
    
    if is_main_process():
        print(f"  Train size: {len(train_dataset)}")
        print(f"  Test size: {len(test_dataset)}")
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True, 
        seed=args.seed
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create SmolLM2-135M model from Hugging Face Hub
    if is_main_process():
        print(f"\nInitializing SmolLM2-135M model from Hugging Face Hub...")
    
    # Use safetensors to avoid torch.load vulnerability issue
    model = AutoModelForSequenceClassification.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        num_labels=4,  # AG News has 4 classes
        token=hf_token,
        use_safetensors=True,  # Use safetensors format for security
        pad_token_id=tokenizer.pad_token_id  # Set pad_token_id in model config
    )
    
    # Also set in model config to ensure it's properly configured
    model.config.pad_token_id = tokenizer.pad_token_id
    
    if is_main_process():
        print(f"  ✓ Loaded SmolLM2-135M from HuggingFaceTB/SmolLM2-135M (safetensors)")
        print(f"  ✓ Model pad_token_id set to: {model.config.pad_token_id}")
    
    # Apply LoRA if requested (must be done before gradient checkpointing)
    if args.use_lora:
        if is_main_process():
            print(f"\nApplying LoRA configuration...")
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # Sequence Classification
            r=args.lora_r,  # LoRA rank
            lora_alpha=args.lora_alpha,  # LoRA alpha for scaling
            lora_dropout=args.lora_dropout,  # Dropout probability
            target_modules=args.lora_target_modules.split(',') if args.lora_target_modules else ["q_proj", "v_proj"],  # Target attention modules
            bias="none",  # Don't train bias parameters
            inference_mode=False,  # Training mode
            # Performance optimizations for LoRA
            init_lora_weights=True,  # Initialize LoRA weights properly
            use_rslora=False,  # Standard LoRA (RSLoRA can be slower)
            use_dora=False,  # Disable DoRA for speed
        )
        
        model = get_peft_model(model, lora_config)
        
        if is_main_process():
            print(f"  ✓ LoRA applied successfully")
            print(f"    - Rank (r): {args.lora_r}")
            print(f"    - Alpha: {args.lora_alpha}")
            print(f"    - Dropout: {args.lora_dropout}")
            print(f"    - Target modules: {lora_config.target_modules}")
            model.print_trainable_parameters()
    
    # Enable gradient checkpointing if requested
    # Note: LoRA + gradient checkpointing + DDP can cause issues, so we disable it when using LoRA
    if args.use_checkpoint and not args.use_lora:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            if is_main_process():
                print(f"  ✓ Gradient checkpointing enabled")
        elif is_main_process():
            print(f"  ℹ Gradient checkpointing not supported for this model")
    elif args.use_checkpoint and args.use_lora:
        if is_main_process():
            print(f"  ℹ Gradient checkpointing disabled (incompatible with LoRA + DDP)")
            print(f"  ℹ LoRA already provides significant memory savings")
    
    model = model.to(device)
    
    # Compile model for better performance (PyTorch 2.0+)
    # This can significantly speed up LoRA training
    if args.use_lora and hasattr(torch, 'compile') and args.compile_model:
        if is_main_process():
            print(f"  ℹ Compiling model with torch.compile for better performance...")
        try:
            model = torch.compile(model, mode='reduce-overhead')
            if is_main_process():
                print(f"  ✓ Model compiled successfully")
        except Exception as e:
            if is_main_process():
                print(f"  ℹ Model compilation failed: {e}, continuing without compilation")
    
    # Wrap with DDP
    # Note: With LoRA properly configured (without gradient checkpointing), we don't need find_unused_parameters
    # Setting it to False improves performance
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  Total parameters: {total_params / 1e6:.1f}M")
        print(f"  Trainable parameters: {trainable_params / 1e6:.1f}M")
        if args.use_lora:
            print(f"  Parameter efficiency: {trainable_params / total_params * 100:.2f}%")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    total_steps = args.epochs * len(train_loader) // args.grad_accum_steps
    warmup_steps = args.warmup_epochs * len(train_loader) // args.grad_accum_steps
    
    def get_lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda)
    
    # AMP scaler
    scaler = GradScaler('cuda', enabled=args.use_amp)
    
    # EMA (disable for LoRA as it adds overhead with minimal benefit)
    # LoRA trains very few parameters, so EMA is less critical
    ema = None
    if not args.use_lora and is_main_process():
        ema = EMA(model.module, decay=args.ema_decay)
    
    # Training configuration
    if is_main_process():
        print(f"\nTraining Configuration:")
        print(f"  - Model: SmolLM2-135M")
        if args.use_lora:
            print(f"  - Fine-tuning method: LoRA (Low-Rank Adaptation)")
            print(f"  - LoRA optimizations: EMA disabled, inference_mode enabled for eval")
        else:
            print(f"  - Fine-tuning method: Full Fine-tuning")
        print(f"  - Epochs: {args.epochs}")
        print(f"  - Batch size per GPU: {args.batch_size}")
        print(f"  - Global batch size: {args.batch_size * world_size * args.grad_accum_steps}")
        print(f"  - Gradient accumulation: {args.grad_accum_steps}")
        print(f"  - Max sequence length: {args.max_length}")
        print(f"  - Learning rate: {args.lr}")
        print(f"  - Weight decay: {args.weight_decay}")
        print(f"  - Warmup epochs: {args.warmup_epochs}")
        if ema is not None:
            print(f"  - EMA decay: {args.ema_decay}")
        else:
            print(f"  - EMA: Disabled")
        print(f"  - Mixed Precision: {'bfloat16' if args.use_amp else 'float32'}")
        print(f"{'='*80}\n")
    
    # Training loop
    train_losses = []
    test_accuracies = []
    learning_rates = []
    epoch_times = []
    
    training_start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        train_sampler.set_epoch(epoch)
        
        total_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            # Mixed precision training
            with autocast('cuda', dtype=torch.bfloat16, enabled=args.use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / args.grad_accum_steps
            
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update EMA
                if ema is not None:
                    ema.update()
                
                # Update learning rate
                scheduler.step()
            
            total_loss += loss.item() * args.grad_accum_steps
        
        avg_loss = total_loss / len(train_loader)
        
        # Synchronize loss across all processes
        if dist.is_initialized():
            loss_tensor = torch.tensor([avg_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        # Evaluate
        test_acc = evaluate_accuracy(
            model.module if hasattr(model, 'module') else model,
            test_loader,
            device,
            dtype=torch.bfloat16
        )
        
        # Synchronize test accuracy
        if dist.is_initialized():
            test_acc_tensor = torch.tensor([test_acc], device=device)
            dist.all_reduce(test_acc_tensor, op=dist.ReduceOp.AVG)
            test_acc = test_acc_tensor.item()
        
        # Record metrics
        if is_main_process():
            train_losses.append(avg_loss)
            test_accuracies.append(test_acc)
            learning_rates.append(optimizer.param_groups[0]['lr'])
            
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_duration)
            
            # Print progress
            if epoch % args.print_freq == 0 or epoch == args.epochs - 1:
                print(f"Epoch {epoch:3d}: Loss={avg_loss:.6f}, Test Acc={test_acc:.4f}, "
                      f"LR={learning_rates[-1]:.8f}, Time={epoch_duration:.2f}s")
        
        # Synchronize
        if dist.is_initialized():
            dist.barrier()
    
    # Final results
    if is_main_process():
        total_training_time = time.time() - training_start_time
        
        print(f"\n{'='*80}")
        print(f"Training completed!")
        print(f"{'='*80}")
        print(f"Total training time: {total_training_time/3600:.2f}h")
        print(f"Average time per epoch: {np.mean(epoch_times):.2f}s")
        print(f"Best test accuracy: {max(test_accuracies):.4f} at epoch {test_accuracies.index(max(test_accuracies))+1}")
        print(f"Final test accuracy: {test_accuracies[-1]:.4f}")
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\nSaving results to {args.output_dir}...")
        
        np.savetxt(os.path.join(args.output_dir, "train_losses.txt"), np.array(train_losses), fmt="%.10f")
        np.savetxt(os.path.join(args.output_dir, "test_accuracies.txt"), np.array(test_accuracies), fmt="%.6f")
        np.savetxt(os.path.join(args.output_dir, "learning_rates.txt"), np.array(learning_rates), fmt="%.10f")
        np.savetxt(os.path.join(args.output_dir, "epoch_times.txt"), np.array(epoch_times), fmt="%.2f")
        
        # Plot training curves
        plot_training_curves(
            train_losses,
            test_accuracies,
            learning_rates,
            epoch_times,
            os.path.join(args.output_dir, "training_curves.png")
        )
        
        print(f"✓ All results saved")
    
    # Cleanup
    cleanup_ddp()


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SmolLM2-135M on AG News with DDP')
    
    # Data
    parser.add_argument('--output-dir', type=str, default='./output_smol_ddp', help='Output directory')
    parser.add_argument('--max-length', type=int, default=512, help='Max sequence length')
    
    # Model
    parser.add_argument('--use-checkpoint', action='store_true', default=True, help='Use gradient checkpointing')
    
    # LoRA configuration
    parser.add_argument('--use-lora', action='store_true', default=False, help='Use LoRA for parameter-efficient fine-tuning')
    parser.add_argument('--lora-r', type=int, default=8, help='LoRA rank (default: 8)')
    parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha scaling parameter (default: 16)')
    parser.add_argument('--lora-dropout', type=float, default=0.1, help='LoRA dropout probability (default: 0.1)')
    parser.add_argument('--lora-target-modules', type=str, default='q_proj,v_proj,k_proj,o_proj', 
                        help='Comma-separated list of target modules for LoRA (default: q_proj,v_proj,k_proj,o_proj)')
    parser.add_argument('--compile-model', action='store_true', default=False, 
                        help='Use torch.compile to optimize LoRA performance (PyTorch 2.0+, may take time to compile)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--grad-accum-steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=1, help='Warmup epochs')
    parser.add_argument('--use-amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--ema-decay', type=float, default=0.9999, help='EMA decay')
    
    # Others
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--print-freq', type=int, default=1, help='Print frequency (epochs)')
    
    # Checkpoint arguments (deprecated - kept for compatibility but not used)
    parser.add_argument('--save-freq', type=int, default=5, help='Save checkpoint every N epochs (deprecated - not used)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from (deprecated - not used)')
    
    args = parser.parse_args()
    
    # Get world size
    world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
    
    # Set MASTER_ADDR and MASTER_PORT
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    
    if 'MASTER_PORT' not in os.environ:
        import socket
        sock = socket.socket()
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()
        os.environ['MASTER_PORT'] = str(port)
    
    if world_size > 1:
        import torch.multiprocessing as mp
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train_worker(0, 1, args)

