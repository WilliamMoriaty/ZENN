#!/usr/bin/env python3
"""
ViT-L/16 training on CIFAR-10 with DDP and performance optimizations
- DistributedDataParallel (DDP)
- Mixed Precision with bfloat16
- Gradient Checkpointing
- channels_last memory format
- TF32 and cudnn optimizations
- Gradient accumulation
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
import timm
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict


# ============================================================================
# DDP Setup and Utilities
# ============================================================================
def setup_ddp(rank, world_size):
    """Initialize DDP"""
    # MASTER_ADDR and MASTER_PORT should be set before calling this function
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
# Position Embedding Interpolation
# ============================================================================
def interpolate_pos_embed(pos_embed, orig_size=14, new_size=24):
    """
    Interpolate position embeddings for different image sizes
    
    Args:
        pos_embed: (1, N+1, dim) where N = orig_size * orig_size
        orig_size: original grid size (14 for 224x224 with patch 16)
        new_size: new grid size (24 for 384x384 with patch 16)
    
    Returns:
        Interpolated position embedding (1, M+1, dim) where M = new_size * new_size
    """
    # Separate class token and position embeddings
    cls_pos_embed = pos_embed[:, 0:1, :]  # (1, 1, dim)
    patch_pos_embed = pos_embed[:, 1:, :]  # (1, N, dim)
    
    dim = patch_pos_embed.shape[-1]
    
    # Reshape to 2D grid
    patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # (1, dim, orig_size, orig_size)
    
    # Interpolate
    patch_pos_embed = torch.nn.functional.interpolate(
        patch_pos_embed,
        size=(new_size, new_size),
        mode='bicubic',
        align_corners=False
    )
    
    # Reshape back
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)  # (1, new_size, new_size, dim)
    patch_pos_embed = patch_pos_embed.reshape(1, new_size * new_size, dim)
    
    # Concatenate class token
    new_pos_embed = torch.cat([cls_pos_embed, patch_pos_embed], dim=1)
    
    return new_pos_embed


# ============================================================================
# Load Google ViT NPZ Weights
# ============================================================================
def load_npz_weights(model, npz_path, num_classes=10):
    """
    Load Google's official ViT pretrained weights from .npz file
    
    Args:
        model: timm ViT model
        npz_path: path to .npz file
        num_classes: number of output classes (for classifier head)
    """
    if is_main_process():
        print(f"\nLoading weights from {npz_path}...")
    
    # Load .npz file
    npz_weights = np.load(npz_path)
    
    if is_main_process():
        print(f"  Found {len(npz_weights.files)} weight arrays in .npz file")
    
    # Get model state dict
    model_state = model.state_dict()
    new_state = OrderedDict()
    
    # Mapping between Google ViT naming and timm naming
    # Google ViT uses: Transformer/encoderblock_X/...
    # timm uses: blocks.X...
    
    try:
        # Load patch embedding
        if 'embedding/kernel' in npz_weights:
            # Google: (patch_h, patch_w, in_channels, embed_dim)
            # timm: (embed_dim, in_channels, patch_h, patch_w)
            patch_embed_weight = npz_weights['embedding/kernel']
            patch_embed_weight = torch.from_numpy(patch_embed_weight).permute(3, 2, 0, 1)
            new_state['patch_embed.proj.weight'] = patch_embed_weight
            if is_main_process():
                print("  ✓ Loaded patch embedding")
        
        if 'embedding/bias' in npz_weights:
            new_state['patch_embed.proj.bias'] = torch.from_numpy(npz_weights['embedding/bias'])
        
        # Load class token
        if 'cls' in npz_weights:
            # Google shape is already (1, 1, 1024), no need to unsqueeze
            new_state['cls_token'] = torch.from_numpy(npz_weights['cls'])
            if is_main_process():
                print("  ✓ Loaded class token")
        
        # Load positional embedding
        if 'Transformer/posembed_input/pos_embedding' in npz_weights:
            pos_embed = torch.from_numpy(npz_weights['Transformer/posembed_input/pos_embedding'])
            # Use 224x224 resolution (no interpolation needed)
            # Original: 224/16 = 14 patches
            if is_main_process():
                print(f"  ✓ Loaded positional embedding (shape: {pos_embed.shape})")
            new_state['pos_embed'] = pos_embed
        
        # Load transformer blocks
        num_layers = 24  # ViT-L has 24 layers
        for layer_idx in range(num_layers):
            prefix_google = f'Transformer/encoderblock_{layer_idx}'
            prefix_timm = f'blocks.{layer_idx}'
            
            # Attention layer norm
            if f'{prefix_google}/LayerNorm_0/scale' in npz_weights:
                new_state[f'{prefix_timm}.norm1.weight'] = torch.from_numpy(
                    npz_weights[f'{prefix_google}/LayerNorm_0/scale'])
            if f'{prefix_google}/LayerNorm_0/bias' in npz_weights:
                new_state[f'{prefix_timm}.norm1.bias'] = torch.from_numpy(
                    npz_weights[f'{prefix_google}/LayerNorm_0/bias'])
            
            # MLP layer norm
            if f'{prefix_google}/LayerNorm_2/scale' in npz_weights:
                new_state[f'{prefix_timm}.norm2.weight'] = torch.from_numpy(
                    npz_weights[f'{prefix_google}/LayerNorm_2/scale'])
            if f'{prefix_google}/LayerNorm_2/bias' in npz_weights:
                new_state[f'{prefix_timm}.norm2.bias'] = torch.from_numpy(
                    npz_weights[f'{prefix_google}/LayerNorm_2/bias'])
            
            # Attention QKV
            if f'{prefix_google}/MultiHeadDotProductAttention_1/query/kernel' in npz_weights:
                # Google shape: (embed_dim, num_heads, head_dim)
                q_weight = npz_weights[f'{prefix_google}/MultiHeadDotProductAttention_1/query/kernel']
                k_weight = npz_weights[f'{prefix_google}/MultiHeadDotProductAttention_1/key/kernel']
                v_weight = npz_weights[f'{prefix_google}/MultiHeadDotProductAttention_1/value/kernel']
                q_bias = npz_weights[f'{prefix_google}/MultiHeadDotProductAttention_1/query/bias']
                k_bias = npz_weights[f'{prefix_google}/MultiHeadDotProductAttention_1/key/bias']
                v_bias = npz_weights[f'{prefix_google}/MultiHeadDotProductAttention_1/value/bias']
                
                # Reshape from (embed_dim, num_heads, head_dim) to (embed_dim, embed_dim)
                q_weight = q_weight.reshape(q_weight.shape[0], -1)
                k_weight = k_weight.reshape(k_weight.shape[0], -1)
                v_weight = v_weight.reshape(v_weight.shape[0], -1)
                q_bias = q_bias.reshape(-1)
                k_bias = k_bias.reshape(-1)
                v_bias = v_bias.reshape(-1)
                
                # Concatenate Q, K, V along the output dimension
                qkv_weight = np.concatenate([q_weight, k_weight, v_weight], axis=1)
                qkv_bias = np.concatenate([q_bias, k_bias, v_bias], axis=0)
                
                # timm expects: (out_features, in_features)
                new_state[f'{prefix_timm}.attn.qkv.weight'] = torch.from_numpy(qkv_weight.T)
                new_state[f'{prefix_timm}.attn.qkv.bias'] = torch.from_numpy(qkv_bias)
            
            # Attention output projection
            if f'{prefix_google}/MultiHeadDotProductAttention_1/out/kernel' in npz_weights:
                # Google shape: (num_heads, head_dim, embed_dim)
                out_weight = npz_weights[f'{prefix_google}/MultiHeadDotProductAttention_1/out/kernel']
                out_bias = npz_weights[f'{prefix_google}/MultiHeadDotProductAttention_1/out/bias']
                # Reshape to (embed_dim, embed_dim)
                out_weight = out_weight.reshape(-1, out_weight.shape[-1])
                # timm expects: (out_features, in_features) so need transpose
                new_state[f'{prefix_timm}.attn.proj.weight'] = torch.from_numpy(out_weight.T)
                new_state[f'{prefix_timm}.attn.proj.bias'] = torch.from_numpy(out_bias)
            
            # MLP layers
            if f'{prefix_google}/MlpBlock_3/Dense_0/kernel' in npz_weights:
                new_state[f'{prefix_timm}.mlp.fc1.weight'] = torch.from_numpy(
                    npz_weights[f'{prefix_google}/MlpBlock_3/Dense_0/kernel']).T
            if f'{prefix_google}/MlpBlock_3/Dense_0/bias' in npz_weights:
                new_state[f'{prefix_timm}.mlp.fc1.bias'] = torch.from_numpy(
                    npz_weights[f'{prefix_google}/MlpBlock_3/Dense_0/bias'])
            
            if f'{prefix_google}/MlpBlock_3/Dense_1/kernel' in npz_weights:
                new_state[f'{prefix_timm}.mlp.fc2.weight'] = torch.from_numpy(
                    npz_weights[f'{prefix_google}/MlpBlock_3/Dense_1/kernel']).T
            if f'{prefix_google}/MlpBlock_3/Dense_1/bias' in npz_weights:
                new_state[f'{prefix_timm}.mlp.fc2.bias'] = torch.from_numpy(
                    npz_weights[f'{prefix_google}/MlpBlock_3/Dense_1/bias'])
        
        if is_main_process():
            print(f"  ✓ Loaded {num_layers} transformer blocks")
        
        # Load final layer norm
        if 'Transformer/encoder_norm/scale' in npz_weights:
            new_state['norm.weight'] = torch.from_numpy(npz_weights['Transformer/encoder_norm/scale'])
        if 'Transformer/encoder_norm/bias' in npz_weights:
            new_state['norm.bias'] = torch.from_numpy(npz_weights['Transformer/encoder_norm/bias'])
            if is_main_process():
                print("  ✓ Loaded final layer norm")
        
        # Note: We skip loading the classifier head since we're training for CIFAR-10
        # The head will be randomly initialized for num_classes=10
        
        # Load weights into model
        missing_keys, unexpected_keys = model.load_state_dict(new_state, strict=False)
        
        if is_main_process():
            print(f"\n  Successfully loaded pretrained weights from .npz file")
            if missing_keys:
                print(f"  Missing keys (will be randomly initialized): {len(missing_keys)}")
                # Only show first few missing keys
                for key in missing_keys[:5]:
                    print(f"    - {key}")
                if len(missing_keys) > 5:
                    print(f"    ... and {len(missing_keys) - 5} more")
            if unexpected_keys:
                print(f"  Unexpected keys: {len(unexpected_keys)}")
        
        return True
        
    except Exception as e:
        if is_main_process():
            print(f"  ✗ Error loading .npz weights: {e}")
            print(f"  Falling back to timm pretrained weights...")
        return False


# ============================================================================
# Model Definition with Gradient Checkpointing
# ============================================================================
class ViT_L_16_CIFAR(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, drop_path_rate=0.1, use_checkpoint=True, 
                 npz_path=None):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        if pretrained:
            # 首先创建模型（不加载预训练权重）
            # 使用 img_size=224（原始预训练分辨率）
            self.model = timm.create_model(
                'vit_large_patch16_224',
                pretrained=False,
                num_classes=num_classes,
                drop_path_rate=drop_path_rate,
                img_size=224
            )
            
            # 尝试加载 Google 官方的 .npz 权重
            loaded_npz = False
            if npz_path and os.path.exists(npz_path):
                if is_main_process():
                    print(f"Loading ViT-L/16 from Google's official .npz weights...")
                loaded_npz = load_npz_weights(self.model, npz_path, num_classes)
                
                if loaded_npz and is_main_process():
                    print(f"✓ Loaded Google's official pretrained weights")
                    print(f"  Source: {npz_path}")
                    print(f"  Pretrained on: ImageNet-21k (14M images)")
                    print(f"  Input size: 224x224")
                    print(f"  Params: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")
            
            # 如果 .npz 加载失败，回退到 timm 预训练权重
            if not loaded_npz:
                if is_main_process():
                    print("Loading ViT-L/16 with pretrained weights from timm...")
                
                # 尝试加载预训练权重
                # 优先级: ImageNet-21k (14M images) > ImageNet-1k (1.3M images)
                try:
                    # 尝试 ImageNet-21k 预训练
                    self.model = timm.create_model(
                        'vit_large_patch16_224.augreg_in21k_ft_in1k',  # IN21K 预训练 + IN1K 微调
                        pretrained=True,
                        num_classes=num_classes,
                        drop_path_rate=drop_path_rate,
                        img_size=224
                    )
                    pretrain_dataset = "ImageNet-21k (14M images)"
                    model_name = "vit_large_patch16_224.augreg_in21k_ft_in1k"
                except Exception as e:
                    if is_main_process():
                        print(f"  ImageNet-21k weights not available, trying alternative: {e}")
                    try:
                        # 尝试另一个 ImageNet-21k 版本
                        self.model = timm.create_model(
                            'vit_large_patch16_224.orig_in21k_ft_in1k',
                            pretrained=True,
                            num_classes=num_classes,
                            drop_path_rate=drop_path_rate,
                            img_size=224
                        )
                        pretrain_dataset = "ImageNet-21k (14M images)"
                        model_name = "vit_large_patch16_224.orig_in21k_ft_in1k"
                    except Exception:
                        # 回退到 ImageNet-1k
                        if is_main_process():
                            print("  ImageNet-21k weights not available, using ImageNet-1k")
                        self.model = timm.create_model(
                            'vit_large_patch16_224',
                            pretrained=True,
                            num_classes=num_classes,
                            drop_path_rate=drop_path_rate,
                            img_size=224
                        )
                        pretrain_dataset = "ImageNet-1k (1.3M images)"
                        model_name = "vit_large_patch16_224"
                
                if is_main_process():
                    print(f"✓ Loaded pretrained weights (ViT-L/16)")
                    print(f"  Model: {model_name}")
                    print(f"  Pretrained on: {pretrain_dataset}")
                    print(f"  Input size: 224x224")
                    print(f"  Params: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")
        else:
            if is_main_process():
                print("Creating ViT-L/16 from scratch (no pretrained weights)...")
            
            self.model = timm.create_model(
                'vit_large_patch16_224',  # 使用标准 ViT-L/16 架构
                pretrained=False,
                num_classes=num_classes,
                drop_path_rate=drop_path_rate,
                img_size=224
            )
            
            if is_main_process():
                print("✓ Model initialized from scratch")
                print(f"  Input size: 224x224")
        
        # Enable gradient checkpointing using timm's built-in method
        if use_checkpoint:
            if hasattr(self.model, 'set_grad_checkpointing'):
                self.model.set_grad_checkpointing(enable=True)
                if is_main_process():
                    print("✓ Gradient checkpointing enabled")
            else:
                if is_main_process():
                    print("⚠ Warning: Gradient checkpointing not available for this model")

    def forward(self, x):
        return self.model(x)


# ============================================================================
# Data Augmentation Functions
# ============================================================================
def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=0.1):
    """CutMix data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    """Generate random bounding box for CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calculate Mixup/CutMix loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cross_entropy_loss_with_label_smoothing(logits, targets, smoothing=0.1):
    """Cross-entropy loss with Label Smoothing"""
    confidence = 1.0 - smoothing
    log_probs = F.log_softmax(logits, dim=1)
    nll_loss = -log_probs.gather(dim=1, index=targets.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -log_probs.mean(dim=1)
    loss = confidence * nll_loss + smoothing * smooth_loss
    return loss.mean()


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


# ============================================================================
# Evaluation Function
# ============================================================================
def evaluate_accuracy(model, loader, device, dtype=torch.bfloat16):
    """Evaluate model accuracy"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # Convert to channels_last
            x = x.to(memory_format=torch.channels_last)
            
            with autocast('cuda', dtype=dtype):
                pred = model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    return correct / total


# ============================================================================
# Plotting Function (only on main process)
# ============================================================================
def plot_training_curves(train_losses, test_accuracies, learning_rates, epoch_times, save_path):
    """Plot and save training curves"""
    if not is_main_process():
        return
    
    epochs_range = np.arange(1, len(train_losses) + 1)
    fig = plt.figure(figsize=(18, 10))
    
    # First row
    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, train_losses, marker='o', markersize=3, label="Train Loss", color='blue')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss Curve", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
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
    
    # Time statistics
    plt.subplot(2, 3, 4)
    plt.plot(epochs_range, epoch_times, marker='o', markersize=2, color='purple', label="Epoch Time")
    plt.axhline(y=np.mean(epoch_times), color='r', linestyle='--', label=f'Mean: {np.mean(epoch_times):.2f}s')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.title("Training Time per Epoch", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 3, 5)
    if len(epoch_times) > 10:
        window_size = min(10, len(epoch_times) // 10)
        moving_avg = np.convolve(epoch_times, np.ones(window_size)/window_size, mode='valid')
        plt.plot(epochs_range[:len(moving_avg)], moving_avg, marker='s', markersize=2, 
                 color='darkblue', label=f'Moving Avg (window={window_size})')
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Time (seconds)", fontsize=12)
        plt.title("Smoothed Training Time", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # Training summary - Combined metrics
    plt.subplot(2, 3, 6)
    info_text = f"Training Summary\n\n"
    info_text += f"Dataset: CIFAR-10\n"
    info_text += f"Model: ViT-L/16\n\n"
    info_text += f"Results:\n"
    info_text += f"  Epochs: {len(train_losses)}\n"
    info_text += f"  Final Loss: {train_losses[-1]:.6f}\n"
    info_text += f"  Best Test Acc: {max(test_accuracies):.4f}\n"
    info_text += f"  Final Test Acc: {test_accuracies[-1]:.4f}\n\n"
    info_text += f"Training Time:\n"
    info_text += f"  Avg/Epoch: {np.mean(epoch_times):.2f}s\n"
    info_text += f"  Total Time: {sum(epoch_times)/3600:.2f}h\n\n"
    info_text += f"Config:\n"
    info_text += f"  DDP + bfloat16\n"
    info_text += f"  Grad Checkpoint\n"
    info_text += f"  Global BS: 512\n"
    
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
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    
    # Performance optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device = torch.device(f"cuda:{rank}")
    
    if is_main_process():
        print(f"Using {world_size} GPU(s) with DDP")
        for i in range(world_size):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"\nPerformance Optimizations:")
        print(f"  - DistributedDataParallel (DDP)")
        print(f"  - Mixed Precision: bfloat16")
        print(f"  - Memory Format: channels_last")
        print(f"  - Gradient Checkpointing: Enabled")
        print(f"  - TF32: Enabled")
        print(f"  - cudnn.benchmark: True")
    
    # Data transforms
    # Using 224x224 input (standard ViT-L/16 resolution)
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                             std=[0.5, 0.5, 0.5]),  # timm 默认使用 [-1, 1] 归一化
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                             std=[0.5, 0.5, 0.5]),
    ])
    
    # Load datasets
    if is_main_process():
        print("\nLoading datasets...")
    
    # Use full training set (no validation split)
    train_dataset = CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(root=args.data_dir, train=False, download=True, transform=eval_transform)
    
    if is_main_process():
        print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        num_workers=args.num_workers, 
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    # Create model
    if is_main_process():
        print("\nInitializing model...")
    
    model = ViT_L_16_CIFAR(
        pretrained=args.pretrained, 
        drop_path_rate=args.drop_path_rate,
        use_checkpoint=args.use_checkpoint,
        npz_path=args.npz_path
    )
    
    # Convert to channels_last memory format
    model = model.to(device, memory_format=torch.channels_last)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # Optimizer and scheduler
    # Using AdamW optimizer (better for Vision Transformers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, 
                                  betas=(0.9, 0.999), weight_decay=args.weight_decay)
    
    total_steps = args.epochs * len(train_loader) // args.grad_accum_steps
    warmup_steps = args.warmup_epochs * len(train_loader) // args.grad_accum_steps
    
    def get_lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda)
    
    # AMP scaler with bfloat16
    scaler = GradScaler('cuda', enabled=args.use_amp)
    
    # EMA (only on main process)
    # Following ViT paper: Polyak averaging with decay=0.9999
    ema = EMA(model.module, decay=args.ema_decay) if is_main_process() else None
    
    # Training configuration
    if is_main_process():
        print("\nTraining Configuration:")
        print(f"  - Model: ViT-L/16 (DDP)")
        print(f"  - Epochs: {args.epochs}")
        print(f"  - Batch size per GPU: {args.batch_size}")
        print(f"  - Global batch size: {args.batch_size * world_size * args.grad_accum_steps}")
        print(f"  - Gradient accumulation steps: {args.grad_accum_steps}")
        print(f"  - Learning rate: {args.lr}")
        print(f"  - Weight decay: {args.weight_decay}")
        print(f"  - Optimizer: AdamW")
        print(f"  - Data Augmentation: RandAug + Mixup({args.mixup_alpha}) + CutMix({args.cutmix_alpha})")
        print(f"  - Label Smoothing: {args.label_smoothing}")
        print(f"  - Regularization: DropPath({args.drop_path_rate}) + EMA({args.ema_decay})")
        print(f"  - Learning Rate: Warm-up {args.warmup_epochs} epochs + Cosine Decay")
        print(f"  - Mixed Precision: {'bfloat16' if args.use_amp else 'float32'}")
    
    # Training loop
    train_losses = []
    test_accuracies = []
    learning_rates = []
    epoch_times = []
    
    # Record training start time
    training_start_time = time.time()
    training_start_datetime = datetime.now()
    
    if is_main_process():
        print(f"\nStarting training at {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}...")
        print("=" * 80)
    
    # GPU memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(rank)
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        train_sampler.set_epoch(epoch)
        
        total_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # Convert to channels_last
            x = x.to(memory_format=torch.channels_last)
            
            # Randomly apply Mixup or CutMix
            r = np.random.rand()
            if r < args.mixup_prob:
                x, y_a, y_b, lam = mixup_data(x, y, alpha=args.mixup_alpha)
                use_mixup = True
            elif r < args.mixup_prob + args.cutmix_prob:
                x, y_a, y_b, lam = cutmix_data(x, y, alpha=args.cutmix_alpha)
                use_mixup = True
            else:
                use_mixup = False
            
            # Mixed precision training with bfloat16
            with autocast('cuda', dtype=torch.bfloat16, enabled=args.use_amp):
                output = model(x)
                if use_mixup:
                    loss_func = nn.CrossEntropyLoss(reduction='mean')
                    loss = mixup_criterion(loss_func, output, y_a, y_b, lam)
                else:
                    loss = cross_entropy_loss_with_label_smoothing(output, y, smoothing=args.label_smoothing)
                
                # Scale loss for gradient accumulation
                loss = loss / args.grad_accum_steps
            
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update EMA (only on main process)
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
        
        # Evaluate test accuracy after each epoch
        test_acc = evaluate_accuracy(model.module if hasattr(model, 'module') else model, 
                                     test_loader, device, dtype=torch.bfloat16)
        
        # Synchronize test accuracy across all processes
        if dist.is_initialized():
            test_acc_tensor = torch.tensor([test_acc], device=device)
            dist.all_reduce(test_acc_tensor, op=dist.ReduceOp.AVG)
            test_acc = test_acc_tensor.item()
        
        # Record metrics (only on main process)
        if is_main_process():
            train_losses.append(avg_loss)
            test_accuracies.append(test_acc)
            learning_rates.append(optimizer.param_groups[0]['lr'])
            
            # Record epoch time
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_duration)
            
            # Calculate ETA
            avg_epoch_time = np.mean(epoch_times)
            remaining_epochs = args.epochs - epoch - 1
            eta_seconds = avg_epoch_time * remaining_epochs
            eta = timedelta(seconds=int(eta_seconds))
            
            # Print progress
            if epoch % args.print_freq == 0 or epoch == args.epochs - 1:
                print(f"Epoch {epoch:3d}: Loss={avg_loss:.6f}, Test Acc={test_acc:.4f}, "
                      f"LR={learning_rates[-1]:.8f}, Time={epoch_duration:.2f}s, ETA={eta}")
        
        # Synchronize all processes
        if dist.is_initialized():
            dist.barrier()
    
    # Final results (only on main process)
    if is_main_process():
        # Calculate total training time
        training_end_time = time.time()
        training_end_datetime = datetime.now()
        total_training_time = training_end_time - training_start_time
        total_training_timedelta = timedelta(seconds=int(total_training_time))
        
        print("\n" + "=" * 80)
        print("Training completed!")
        print("=" * 80)
        print(f"Start time: {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time: {training_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total training time: {total_training_timedelta} ({total_training_time:.2f}s)")
        print(f"Average time per epoch: {np.mean(epoch_times):.2f}s")
        print(f"Min/Max epoch time: {np.min(epoch_times):.2f}s / {np.max(epoch_times):.2f}s")
        print("\nTraining Performance:")
        print(f"Best test accuracy: {max(test_accuracies):.4f} at epoch {test_accuracies.index(max(test_accuracies))+1}")
        print(f"Final test accuracy: {test_accuracies[-1]:.4f}")
        print(f"Final train loss: {train_losses[-1]:.6f}")
        
        # GPU statistics
        if torch.cuda.is_available():
            print("\nGPU Statistics (Rank 0):")
            print("-" * 80)
            gpu_name = torch.cuda.get_device_name(rank)
            max_memory = torch.cuda.max_memory_allocated(rank) / 1024**3
            total_memory = torch.cuda.get_device_properties(rank).total_memory / 1024**3
            utilization = (max_memory / total_memory) * 100
            print(f"  GPU {rank} ({gpu_name}):")
            print(f"    Max memory used: {max_memory:.2f} GB / {total_memory:.2f} GB ({utilization:.1f}%)")
            print("-" * 80)
        
        # Final test evaluation
        print("\nEvaluating on test set...")
        test_acc = evaluate_accuracy(model.module, test_loader, device, dtype=torch.bfloat16)
        print(f"Original model test accuracy: {test_acc:.4f}")
        
        if ema is not None:
            ema.apply_shadow()
            ema_test_acc = evaluate_accuracy(model.module, test_loader, device, dtype=torch.bfloat16)
            ema.restore()
            print(f"EMA model test accuracy: {ema_test_acc:.4f}")
        else:
            ema_test_acc = test_acc
        
        # Save results
        print("\nSaving results...")
        np.savetxt(os.path.join(args.output_dir, "train_losses.txt"), np.array(train_losses), fmt="%.10f")
        np.savetxt(os.path.join(args.output_dir, "test_accuracies.txt"), np.array(test_accuracies), fmt="%.6f")
        np.savetxt(os.path.join(args.output_dir, "learning_rates.txt"), np.array(learning_rates), fmt="%.10f")
        np.savetxt(os.path.join(args.output_dir, "epoch_times.txt"), np.array(epoch_times), fmt="%.2f")
        
        # Save combined metrics in CSV format for easy analysis
        with open(os.path.join(args.output_dir, "training_metrics.csv"), "w") as f:
            f.write("epoch,train_loss,test_accuracy,learning_rate,epoch_time\n")
            for i in range(len(train_losses)):
                f.write(f"{i+1},{train_losses[i]:.10f},{test_accuracies[i]:.6f},"
                       f"{learning_rates[i]:.10f},{epoch_times[i]:.2f}\n")
        
        # Save summary
        with open(os.path.join(args.output_dir, "results_summary.txt"), "w") as f:
            f.write(f"ViT-L/16 on CIFAR-10 (DDP Optimized)\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Training Configuration:\n")
            f.write(f"  - Model: ViT-L/16 (pretrained)\n")
            f.write(f"  - Epochs: {args.epochs}\n")
            f.write(f"  - Batch size per GPU: {args.batch_size}\n")
            f.write(f"  - Global batch size: {args.batch_size * world_size * args.grad_accum_steps}\n")
            f.write(f"  - Gradient accumulation: {args.grad_accum_steps}\n")
            f.write(f"  - Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})\n")
            f.write(f"  - Data Augmentation: RandAug + Mixup({args.mixup_alpha}) + CutMix({args.cutmix_alpha})\n")
            f.write(f"  - Label Smoothing: {args.label_smoothing}\n")
            f.write(f"  - Regularization: DropPath({args.drop_path_rate}) + EMA({args.ema_decay})\n")
            f.write(f"  - LR Schedule: Warm-up {args.warmup_epochs} epochs + Cosine Decay\n")
            f.write(f"  - Number of GPUs: {world_size}\n")
            f.write(f"  - Mixed Precision: bfloat16\n")
            f.write(f"  - Memory Format: channels_last\n")
            f.write(f"  - Gradient Checkpointing: {args.use_checkpoint}\n")
            f.write(f"  - TF32: Enabled\n\n")
            
            f.write(f"Training Time Statistics:\n")
            f.write(f"  - Start time: {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  - End time: {training_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  - Total training time: {total_training_timedelta} ({total_training_time:.2f}s)\n")
            f.write(f"  - Average time per epoch: {np.mean(epoch_times):.2f}s\n")
            f.write(f"  - Min epoch time: {np.min(epoch_times):.2f}s\n")
            f.write(f"  - Max epoch time: {np.max(epoch_times):.2f}s\n\n")
            
            f.write(f"GPU Statistics (Rank 0):\n")
            f.write(f"  - GPU name: {gpu_name}\n")
            f.write(f"  - Total memory: {total_memory:.2f} GB\n")
            f.write(f"  - Max memory used: {max_memory:.2f} GB ({utilization:.1f}%)\n\n")
            
            f.write(f"Training Results:\n")
            f.write(f"  - Best test accuracy: {max(test_accuracies):.4f} (Epoch {test_accuracies.index(max(test_accuracies))+1})\n")
            f.write(f"  - Final test accuracy: {test_accuracies[-1]:.4f}\n")
            f.write(f"  - Final train loss: {train_losses[-1]:.6f}\n\n")
            
            f.write(f"Final Evaluation Results:\n")
            f.write(f"  - Test accuracy (Original): {test_acc:.4f}\n")
            f.write(f"  - Test accuracy (EMA): {ema_test_acc:.4f}\n")
        
        # Plot training curves
        plot_training_curves(train_losses, test_accuracies, learning_rates, epoch_times,
                             os.path.join(args.output_dir, "training_curves.png"))
        
        print(f"\nAll results saved to {args.output_dir}")
    
    # Cleanup
    cleanup_ddp()


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ViT-L/16 on CIFAR-10 with DDP and optimizations')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='./data', help='Dataset directory')
    parser.add_argument('--output-dir', type=str, default='./output_ddp', help='Output directory')
    
    # Model
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained weights')
    parser.add_argument('--npz-path', type=str, default='./pretrained_models/ViT-L_16.npz', 
                        help='Path to Google ViT .npz pretrained weights')
    parser.add_argument('--drop-path-rate', type=float, default=0.1, help='Drop path rate')
    parser.add_argument('--use-checkpoint', action='store_true', default=True, help='Use gradient checkpointing')
    
    # Training
    parser.add_argument('--epochs', type=int, default=201, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size per GPU (512 total with 8 GPUs)')
    parser.add_argument('--grad-accum-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping value (clip at global norm 1)')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate for AdamW (typical: 0.0001-0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.1, help='Weight decay for AdamW (typical: 0.05-0.1)')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--use-amp', action='store_true', default=True, help='Use automatic mixed precision')
    
    # Data augmentation
    parser.add_argument('--mixup-alpha', type=float, default=0.2, help='Mixup alpha')
    parser.add_argument('--cutmix-alpha', type=float, default=0.1, help='CutMix alpha')
    parser.add_argument('--mixup-prob', type=float, default=0.5, help='Probability to use Mixup')
    parser.add_argument('--cutmix-prob', type=float, default=0.5, help='Probability to use CutMix')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing')
    
    # Regularization
    parser.add_argument('--ema-decay', type=float, default=0.9999, help='EMA decay rate')
    
    # Others
    parser.add_argument('--seed', type=int, default=100, help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers per GPU')
    parser.add_argument('--print-freq', type=int, default=10, help='Print frequency (epochs)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get world size from environment or detect GPUs
    world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
    
    # Set MASTER_ADDR and MASTER_PORT if not already set (for DDP)
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    
    if 'MASTER_PORT' not in os.environ:
        # Find an available port to avoid conflicts
        import socket
        sock = socket.socket()
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()
        os.environ['MASTER_PORT'] = str(port)
        print(f"Using auto-detected port: {port}")
    
    if world_size > 1:
        # Launch DDP training
        import torch.multiprocessing as mp
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        # Single GPU training
        train_worker(0, 1, args)

