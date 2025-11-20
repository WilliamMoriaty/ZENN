#!/usr/bin/env python3
"""
SmolLM2 CoupledModel training on BBC News with DDP, LoRA, and Temperature Learning
- DistributedDataParallel (DDP)
- Mixed Precision with bfloat16
- Coupled Model (E_net and S_net)
- Learnable Temperature Module
- EM Training with Posterior Evaluation
- LoRA for parameter-efficient fine-tuning
"""

import os
import argparse
import time
from datetime import datetime
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

# Try to import PEFT for LoRA support
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. LoRA will be disabled.")


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
# BBC News Dataset Wrapper
# ============================================================================
class BBCNewsDataset(Dataset):
    """
    BBC News dataset wrapper for PyTorch
    5 classes: business, entertainment, politics, sport, tech
    """
    def __init__(self, split, tokenizer, max_length=512, token=None):
        """
        Args:
            split: 'train' or 'test'
            tokenizer: SmolLM2 tokenizer
            max_length: maximum sequence length
            token: HuggingFace token for authentication
        """
        self.dataset = load_dataset('SetFit/bbc-news', split=split, token=token)
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
# Model Definitions
# ============================================================================
class SmolLM2_Classifier(nn.Module):
    """SmolLM2 with optional LoRA for text classification"""
    def __init__(self, model_name, num_classes=5, token=None, pad_token_id=None, use_lora=False, lora_config=None):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            token=token,
            use_safetensors=True,
        )
        
        # Set pad_token_id in model config (critical for batch processing)
        if pad_token_id is not None:
            self.model.config.pad_token_id = pad_token_id
        
        # Apply LoRA if requested
        if use_lora and PEFT_AVAILABLE and lora_config is not None:
            self.model = get_peft_model(self.model, lora_config)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits  # [B, C]


class CoupledModel(nn.Module):
    """
    Coupled Model for text classification with two sub-networks (E_net and S_net)
    Adapted from CoupledModel in ViT implementation for text data
    """
    def __init__(self, model_name, num_classes=5, kb=1.0, rank=0, token=None, pad_token_id=None,
                 use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.1, lora_target_modules=None):
        super().__init__()
        self.C = num_classes
        self.kb = kb
        self.rank = rank
        self.device_E = torch.device(f"cuda:{rank}")
        self.device_S = torch.device(f"cuda:{rank}")
        
        # LoRA configuration
        lora_config = None
        if use_lora and PEFT_AVAILABLE:
            if lora_target_modules is None:
                target_modules = ["q_proj", "v_proj"]
            else:
                target_modules = lora_target_modules.split(',')
            
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
                inference_mode=False,
                init_lora_weights=True,
                use_rslora=False,
                use_dora=False,
            )
        
        # Create two sub-networks: E_net and S_net
        self.E_net = SmolLM2_Classifier(
            model_name, num_classes, token, pad_token_id, use_lora, lora_config
        ).to(self.device_E)
        
        self.S_net = SmolLM2_Classifier(
            model_name, num_classes, token, pad_token_id, use_lora, lora_config
        ).to(self.device_S)
    
    def forward_text(self, input_ids, attention_mask):
        """
        Forward pass to extract E(x) and S(x)
        """
        # E_net forward
        input_ids_E = input_ids.to(self.device_E, non_blocking=True)
        attention_mask_E = attention_mask.to(self.device_E, non_blocking=True)
        
        with torch.cuda.device(self.device_E):
            E_mat = self.E_net(input_ids_E, attention_mask_E)
        
        # S_net forward
        input_ids_S = input_ids.to(self.device_S, non_blocking=True)
        attention_mask_S = attention_mask.to(self.device_S, non_blocking=True)
        
        with torch.cuda.device(self.device_S):
            S_mat = self.S_net(input_ids_S, attention_mask_S)
        
        # Ensure both on device_E
        E_mat = E_mat.to(self.device_E, non_blocking=True).float()
        S_mat = S_mat.to(self.device_E, non_blocking=True).float()
        
        sub_outs = torch.stack([E_mat, S_mat], dim=1)
        return E_mat, S_mat, sub_outs
    
    def _normalize_T(self, T, B, device):
        """
        Normalize temperature T to shape [B, M, 1] for broadcasting
        Supports: scalar, [M], [B,1], [M,1], [B,M]
        """
        if not torch.is_tensor(T):
            T = torch.tensor(T, dtype=torch.float32, device=device)
        
        T = T.to(device)
        if T.dim() == 0:  # scalar -> [1,1,1]
            T = T.view(1, 1, 1)
        elif T.dim() == 1:  # [M] -> [1,M,1]
            T = T.view(1, -1, 1)
        elif T.dim() == 2:
            if T.size(1) == 1:  # [B,1] or [M,1]
                if T.size(0) == B:  # [B,1] -> [B,1,1]
                    T = T.view(B, 1, 1)
                else:  # [M,1] -> [1,M,1]
                    T = T.view(1, -1, 1)
            else:  # [B,M] -> [B,M,1]
                T = T.unsqueeze(-1)
        return T
    
    def forward(self, input_ids, attention_mask, T):
        """
        Forward pass with temperature
        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            T: scalar or tensor, can be [B,1]/[M,1]/[B,M]/[M]
        Returns:
            class_probs: [B, C] probability distribution
            scores: [B, C] unnormalized scores
            sub_outs: [B, 2, C] E/S outputs for debugging
        """
        device = input_ids.device
        eps = 1e-9
        B = input_ids.size(0)
        
        # Get E(x) and S(x)
        E_mat, S_mat, sub_outs = self.forward_text(input_ids, attention_mask)
        S_pos = F.softplus(S_mat)
        
        # Normalize T -> [B, M, 1]
        T = self._normalize_T(T, B, device)
        
        # Broadcast E/S to [B, M, C]
        E_b = E_mat.unsqueeze(1)  # [B,1,C]
        S_b = S_pos.unsqueeze(1)  # [B,1,C]
        
        # Compute scores using energy formula: [B, M, C]
        scores_bmc = -(E_b - T * S_b) / (self.kb * (T + eps)) - (S_b / (100.0 * self.kb)) ** 2
        probs_bmc = F.softmax(scores_bmc, dim=2)  # [B, M, C]
        
        # If single temperature per sample, squeeze; otherwise marginalize
        if scores_bmc.size(1) == 1:
            scores = scores_bmc.squeeze(1)  # [B,C]
            probs = probs_bmc.squeeze(1)  # [B,C]
        else:
            scores = scores_bmc.mean(dim=1)  # [B,C]
            probs = probs_bmc.mean(dim=1)  # [B,C]
        
        return probs, scores, sub_outs


class LearnableTSet(nn.Module):
    """Learnable temperature set module"""
    def __init__(self, K=3, T_min=0.1, T_max=10.0):
        super().__init__()
        self.K = K
        self.T_min = T_min
        self.T_max = T_max
        self.raw_lambdas = nn.Parameter(torch.randn(K))
    
    def forward(self):
        lambdas = torch.sigmoid(self.raw_lambdas)  # [0,1]
        Ts = self.T_min + (self.T_max - self.T_min) * lambdas  # [K]
        return torch.cat([torch.tensor([1.0], device=Ts.device), Ts], dim=0)  # [K+1]


# ============================================================================
# Training Functions
# ============================================================================
def em_train_step_optimized_T(model, input_ids, attention_mask, y_onehot, T_module, 
                               optimizer, scheduler, scaler, grad_clip=1.0, grad_accum_steps=1):
    """
    EM training step with temperature optimization
    E-step: compute posterior q(T|x,y)
    M-step: maximize weighted log-likelihood
    """
    model.train()
    T_grid = T_module()  # [M]
    N, C = y_onehot.shape
    M = T_grid.size(0)
    device = input_ids.device
    
    with autocast('cuda', dtype=torch.bfloat16):
        # Forward to get E_mat and S_mat
        if hasattr(model, 'module'):
            E_mat, S_mat, _ = model.module.forward_text(input_ids, attention_mask)
        else:
            E_mat, S_mat, _ = model.forward_text(input_ids, attention_mask)
        
        S_pos = F.softplus(S_mat)
        
        # Build score matrix [N, M, C]
        if hasattr(model, 'module'):
            T_norm = model.module._normalize_T(T_grid, B=N, device=device)
        else:
            T_norm = model._normalize_T(T_grid, B=N, device=device)
        
        E_b = E_mat.unsqueeze(1)  # [N,1,C]
        S_b = S_pos.unsqueeze(1)  # [N,1,C]
        eps = 1e-9
        kb = model.module.kb if hasattr(model, 'module') else model.kb
        
        scores_bmc = -(E_b - T_norm * S_b) / (kb * (T_norm + eps)) - (S_b / (100.0 * kb)) ** 2
        log_probs_bmc = F.log_softmax(scores_bmc, dim=2)  # [N, M, C]
        
        # Cross-entropy: [N, M]
        ce_bm = -(y_onehot.unsqueeze(1) * log_probs_bmc).sum(dim=2)
        
        # M-step: worst-T weighting with sharpening
        lambda_sharp = 5.0
        qT = torch.softmax(-lambda_sharp * ce_bm, dim=1)
        loss = (qT * ce_bm).sum() / N
        
        # Scale loss for gradient accumulation
        loss = loss / grad_accum_steps
    
    # Backward
    scaler.scale(loss).backward()
    
    return loss.item() * grad_accum_steps, qT.detach()


@torch.no_grad()
def evaluate_accuracy_posterior_labeled(model, loader, device, T_module):
    """
    Evaluate accuracy using posterior q(T|x,y)
    q(T|x,y) ∝ exp(-CZ(p(y|x,T), y))
    """
    model.eval()
    correct = total = 0
    
    T_grid = T_module().detach().to(device)  # [M]
    M = T_grid.size(0)
    
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        y = batch['labels'].to(device)
        
        N = input_ids.size(0)
        
        # Replicate inputs for each temperature
        input_ids_rep = input_ids.unsqueeze(1).repeat(1, M, 1).reshape(N * M, -1)
        attention_mask_rep = attention_mask.unsqueeze(1).repeat(1, M, 1).reshape(N * M, -1)
        T_rep = T_grid.view(1, M, 1).expand(N, M, 1).reshape(N * M, 1)
        
        # Forward
        with autocast('cuda', dtype=torch.bfloat16):
            if hasattr(model, 'module'):
                probs, scores, *_ = model.module(input_ids_rep, attention_mask_rep, T_rep)
            else:
                probs, scores, *_ = model(input_ids_rep, attention_mask_rep, T_rep)
        
        Ccls = probs.size(1)
        probs_nm = probs.view(N, M, Ccls)  # [N, M, C]
        scores_nm = scores.view(N, M, Ccls)  # [N, M, C]
        
        # Compute cross-entropy for each temperature
        log_probs_nm = F.log_softmax(scores_nm, dim=2)  # [N, M, C]
        y_onehot = F.one_hot(y, num_classes=Ccls).float()  # [N, C]
        ce_mat = -(y_onehot.unsqueeze(1) * log_probs_nm).sum(dim=2)  # [N, M]
        
        # E-step: q(T|x,y)
        qT = torch.softmax(-ce_mat, dim=1)  # [N, M]
        
        # Marginalize with posterior weights
        probs_marg_q = (qT.unsqueeze(-1) * probs_nm).sum(dim=1)  # [N, C]
        
        # Predict
        pred = probs_marg_q.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += N
    
    return correct / total


@torch.no_grad()
def posterior_T_labeled_all(model, loader, T_module, device):
    """Compute posterior temperature distribution for entire dataset"""
    model.eval()
    all_qT = []
    all_Tmap = []
    all_idx = []
    
    T_grid = T_module().detach().to(device)
    M = T_grid.size(0)
    
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        y = batch['labels'].to(device)
        
        N = input_ids.size(0)
        
        # Replicate inputs
        input_ids_rep = input_ids.unsqueeze(1).repeat(1, M, 1).reshape(N * M, -1)
        attention_mask_rep = attention_mask.unsqueeze(1).repeat(1, M, 1).reshape(N * M, -1)
        T_rep = T_grid.view(1, M, 1).expand(N, M, 1).reshape(N * M, 1)
        y_rep = y.unsqueeze(1).repeat(1, M).reshape(N * M).long()
        
        # Forward
        with autocast('cuda', dtype=torch.bfloat16):
            if hasattr(model, 'module'):
                probs, scores, *_ = model.module(input_ids_rep, attention_mask_rep, T_rep)
            else:
                probs, scores, *_ = model(input_ids_rep, attention_mask_rep, T_rep)
        
        log_probs = F.log_softmax(scores, dim=1)
        
        # CE loss
        ce_vec = F.nll_loss(log_probs, y_rep, reduction="none")
        ce_mat = ce_vec.view(N, M)
        
        # Posterior q(T|x,y)
        qT = torch.softmax(-ce_mat, dim=1)
        
        # MAP temperature
        idx_map = qT.argmax(dim=1)
        T_map = T_grid.view(-1)[idx_map]
        
        all_qT.append(qT.cpu())
        all_Tmap.append(T_map.cpu())
        all_idx.append(idx_map.cpu())
    
    return torch.cat(all_qT), torch.cat(all_Tmap), torch.cat(all_idx)


# ============================================================================
# Plotting Function
# ============================================================================
def plot_training_curves(train_losses, test_accuracies, T_records, save_path, test_epochs=None):
    """Plot and save training curves"""
    if not is_main_process():
        return
    
    epochs_range = np.arange(1, len(train_losses) + 1)
    
    if test_epochs is None:
        if len(test_accuracies) == len(train_losses):
            test_epochs = epochs_range
        else:
            test_epochs = np.linspace(1, len(train_losses), len(test_accuracies), dtype=int)
    
    fig = plt.figure(figsize=(18, 6))
    
    # Training Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, marker='o', markersize=3, label="Train Loss", color='blue')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss Curve", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Test Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(test_epochs[:len(test_accuracies)], test_accuracies, marker='s', markersize=3, color='green', label="Test Accuracy")
    if test_accuracies:
        best_acc = max(test_accuracies)
        best_idx = test_accuracies.index(best_acc)
        best_epoch = test_epochs[best_idx] if best_idx < len(test_epochs) else best_idx + 1
        plt.axhline(y=best_acc, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_acc:.4f} (Epoch {best_epoch})')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Test Accuracy Curve", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Temperature Evolution
    plt.subplot(1, 3, 3)
    T_records_np = np.array(T_records)
    for i in range(T_records_np.shape[1]):
        plt.plot(epochs_range, T_records_np[:, i], marker='o', markersize=2, label=f'T_{i}')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Temperature", fontsize=12)
    plt.title("Temperature Evolution", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if is_main_process():
        print(f"✓ Training curves saved to {save_path}")


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
        print(f"SmolLM2 CoupledModel Training on BBC News with DDP")
        print(f"{'='*80}")
        print(f"Using {world_size} GPU(s) with DDP")
        for i in range(world_size):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"\nConfiguration:")
        print(f"  Model: {args.model_name}")
        print(f"  Coupled Architecture: E_net + S_net")
        print(f"  Dataset: BBC News (5 classes)")
        print(f"  Use LoRA: {args.use_lora}")
        if args.use_lora:
            print(f"  LoRA Rank: {args.lora_r}")
            print(f"  LoRA Alpha: {args.lora_alpha}")
        print(f"  Batch Size per GPU: {args.batch_size}")
        print(f"  Gradient Accumulation Steps: {args.grad_accum_steps}")
        print(f"  Global Batch Size: {args.batch_size * world_size * args.grad_accum_steps}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Learning Rate (E/S): {args.lr}")
        print(f"  Learning Rate (T): {args.lr_t}")
        print(f"  Weight Decay: {args.weight_decay}")
        print(f"  Warmup Epochs: {args.warmup_epochs}")
        print(f"  Gradient Clip: {args.grad_clip}")
        print(f"  Mixed Precision: bfloat16")
        print(f"  K (Learnable T): {args.K}")
        print(f"  T Range: [{args.T_min}, {args.T_max}]")
        print(f"{'='*80}\n")
    
    # Load tokenizer
    if is_main_process():
        print(f"Loading tokenizer from {args.model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if is_main_process():
        print(f"  ✓ Loaded tokenizer")
        print(f"  ✓ Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    # Load datasets
    if is_main_process():
        print(f"\nLoading BBC News dataset...")
    
    train_dataset = BBCNewsDataset('train', tokenizer, max_length=args.max_length, token=args.hf_token)
    test_dataset = BBCNewsDataset('test', tokenizer, max_length=args.max_length, token=args.hf_token)
    
    if is_main_process():
        print(f"  Train size: {len(train_dataset)}")
        print(f"  Test size: {len(test_dataset)}")
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    
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
    
    # Create model
    if is_main_process():
        print("\nInitializing CoupledModel...")
    
    model = CoupledModel(
        model_name=args.model_name,
        num_classes=5,
        kb=args.kb,
        rank=rank,
        token=args.hf_token,
        pad_token_id=tokenizer.pad_token_id,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules
    ).to(device)
    
    if is_main_process():
        print(f"  ✓ CoupledModel created")
        print(f"  ✓ E_net pad_token_id: {model.E_net.model.config.pad_token_id}")
        print(f"  ✓ S_net pad_token_id: {model.S_net.model.config.pad_token_id}")
    
    T_module = LearnableTSet(K=args.K, T_min=args.T_min, T_max=args.T_max).to(device)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params / 1e6:.1f}M")
        print(f"  Trainable parameters: {trainable_params / 1e6:.1f}M")
        if args.use_lora:
            print(f"  Parameter efficiency: {trainable_params / total_params * 100:.2f}%")
    
    # Optimizer
    params = [
        {"params": model.module.E_net.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": model.module.S_net.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": T_module.parameters(), "lr": args.lr_t, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, betas=(0.9, 0.999))
    
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
    scaler = GradScaler('cuda', enabled=True)
    
    if is_main_process():
        print("✓ Model and optimizer initialized")
        print(f"Starting training...\n")
    
    # Training loop
    train_losses = []
    test_accuracies = []
    test_epochs_list = []
    T_records = []
    
    training_start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        train_sampler.set_epoch(epoch)
        
        # Synchronize T_module parameters
        if dist.is_initialized():
            for param in T_module.parameters():
                dist.broadcast(param.data, src=0)
        
        total_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            y_onehot = F.one_hot(labels, num_classes=5).float()
            
            # EM training step
            loss, qT = em_train_step_optimized_T(
                model, input_ids, attention_mask, y_onehot, T_module,
                optimizer, scheduler, scaler,
                grad_clip=args.grad_clip, grad_accum_steps=args.grad_accum_steps
            )
            total_loss += loss
            
            # Gradient accumulation
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                # Synchronize T_module gradients
                if dist.is_initialized():
                    for param in T_module.parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(T_module.parameters(), args.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                scheduler.step()
        
        # Record
        avg_loss = total_loss / len(train_loader)
        
        # Synchronize loss
        if dist.is_initialized():
            loss_tensor = torch.tensor([avg_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        # Get current temperatures
        with torch.no_grad():
            current_T = T_module().detach().cpu().numpy().flatten()
        
        if is_main_process():
            train_losses.append(avg_loss)
            T_records.append(current_T.copy())
        
        # Evaluate
        if epoch % args.print_freq == 0 or epoch == args.epochs - 1:
            test_acc = evaluate_accuracy_posterior_labeled(model, test_loader, device, T_module)
            
            # Synchronize test accuracy
            if dist.is_initialized():
                test_acc_tensor = torch.tensor([test_acc], device=device)
                dist.all_reduce(test_acc_tensor, op=dist.ReduceOp.AVG)
                test_acc = test_acc_tensor.item()
            
            if is_main_process():
                test_accuracies.append(test_acc)
                test_epochs_list.append(epoch + 1)
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch:3d}: Loss = {avg_loss:.6f}, Test Acc = {test_acc:.6f}, "
                      f"Time = {epoch_time:.2f}s, T = {current_T}")
    
    if is_main_process():
        total_time = time.time() - training_start_time
        print(f"\nTraining complete! Total time: {total_time/3600:.2f}h")
        
        # Compute posterior distributions
        print("\nComputing posterior T distributions...")
        qT_all, Tmap_all, idx_all = posterior_T_labeled_all(model, test_loader, T_module, device)
        
        # Temperature distribution
        max_idx = torch.argmax(qT_all, dim=1)
        qT_onehot = F.one_hot(max_idx, num_classes=qT_all.shape[1]).float()
        counts = qT_onehot.sum(dim=0)
        freqs = counts / qT_onehot.shape[0]
        
        print(f"Temperature distribution:")
        print(f"  Counts: {counts.numpy()}")
        print(f"  Frequencies: {freqs.numpy()}")
        
        # Save results
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving results to {output_dir}...")
        np.savetxt(os.path.join(output_dir, "train_losses.txt"), np.array(train_losses), fmt="%.10f")
        np.savetxt(os.path.join(output_dir, "test_accuracies.txt"), np.array(test_accuracies), fmt="%.10f")
        np.savetxt(os.path.join(output_dir, "test_epochs.txt"), np.array(test_epochs_list), fmt="%d")
        np.savetxt(os.path.join(output_dir, "T_records.txt"), np.array(T_records), fmt="%.10f")
        np.savetxt(os.path.join(output_dir, "freqs.txt"), freqs.numpy(), fmt="%.10f")
        np.savetxt(os.path.join(output_dir, "qT_all.txt"), qT_all.numpy(), fmt="%.4f")
        
        # Plot
        plot_training_curves(train_losses, test_accuracies, T_records,
                           os.path.join(output_dir, "training_curves.png"),
                           test_epochs=test_epochs_list)
        
        # Save model
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'T_module_state_dict': T_module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': args.epochs,
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'test_epochs': test_epochs_list,
        }, os.path.join(output_dir, "checkpoint_final.pth"))
        
        print("✓ All results saved successfully")
        print(f"\nFinal Results:")
        print(f"  Best Test Accuracy: {max(test_accuracies):.4f}")
        print(f"  Final Test Accuracy: {test_accuracies[-1]:.4f}")
        print(f"  Final Loss: {train_losses[-1]:.6f}")
    
    # Cleanup
    cleanup_ddp()


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SmolLM2 CoupledModel Training with DDP')
    
    # Data arguments
    parser.add_argument('--output-dir', type=str, default='./output_coupled_ddp', help='Output directory')
    parser.add_argument('--max-length', type=int, default=512, help='Max sequence length')
    
    # Model arguments
    parser.add_argument('--model-name', type=str, default='HuggingFaceTB/SmolLM2-135M', help='Model name')
    parser.add_argument('--hf-token', type=str, default='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', help='HuggingFace token')
    parser.add_argument('--kb', type=float, default=1.0, help='Boltzmann constant')
    
    # LoRA arguments
    parser.add_argument('--use-lora', action='store_true', default=False, help='Use LoRA')
    parser.add_argument('--lora-r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.1, help='LoRA dropout')
    parser.add_argument('--lora-target-modules', type=str, default='', help='LoRA target modules')
    
    # Temperature arguments
    parser.add_argument('--K', type=int, default=4, help='Number of learnable temperatures')
    parser.add_argument('--T-min', type=float, default=0.1, help='Minimum temperature')
    parser.add_argument('--T-max', type=float, default=10.0, help='Maximum temperature')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--grad-accum-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for E/S nets')
    parser.add_argument('--lr-t', type=float, default=1e-3, help='Learning rate for T module')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=1, help='Warmup epochs')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--print-freq', type=int, default=1, help='Print frequency')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
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

