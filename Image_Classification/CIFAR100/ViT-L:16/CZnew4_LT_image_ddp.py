#!/usr/bin/env python3
"""
ViT-L/16 CoupledModel training on CIFAR-100 with DDP and performance optimizations
- DistributedDataParallel (DDP)
- Mixed Precision with bfloat16
- Temperature Learning
- Multi-GPU training
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
import torch.multiprocessing as mp
import torchvision.models as models
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timm
from collections import OrderedDict


# ============================================================================
# Load Google ViT NPZ Weights
# ============================================================================
def load_npz_weights(model, npz_path, num_classes=100):
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
    
    try:
        # Load patch embedding
        if 'embedding/kernel' in npz_weights:
            patch_embed_weight = npz_weights['embedding/kernel']
            patch_embed_weight = torch.from_numpy(patch_embed_weight).permute(3, 2, 0, 1)
            new_state['patch_embed.proj.weight'] = patch_embed_weight
            if is_main_process():
                print("  ✓ Loaded patch embedding")
        
        if 'embedding/bias' in npz_weights:
            new_state['patch_embed.proj.bias'] = torch.from_numpy(npz_weights['embedding/bias'])
        
        # Load class token
        if 'cls' in npz_weights:
            new_state['cls_token'] = torch.from_numpy(npz_weights['cls'])
            if is_main_process():
                print("  ✓ Loaded class token")
        
        # Load positional embedding
        if 'Transformer/posembed_input/pos_embedding' in npz_weights:
            pos_embed = torch.from_numpy(npz_weights['Transformer/posembed_input/pos_embedding'])
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
                
                # Concatenate Q, K, V
                qkv_weight = np.concatenate([q_weight, k_weight, v_weight], axis=1)
                qkv_bias = np.concatenate([q_bias, k_bias, v_bias], axis=0)
                
                new_state[f'{prefix_timm}.attn.qkv.weight'] = torch.from_numpy(qkv_weight.T)
                new_state[f'{prefix_timm}.attn.qkv.bias'] = torch.from_numpy(qkv_bias)
            
            # Attention output projection
            if f'{prefix_google}/MultiHeadDotProductAttention_1/out/kernel' in npz_weights:
                out_weight = npz_weights[f'{prefix_google}/MultiHeadDotProductAttention_1/out/kernel']
                out_bias = npz_weights[f'{prefix_google}/MultiHeadDotProductAttention_1/out/bias']
                out_weight = out_weight.reshape(-1, out_weight.shape[-1])
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
        
        # Load weights into model
        missing_keys, unexpected_keys = model.load_state_dict(new_state, strict=False)
        
        if is_main_process():
            print(f"  Missing keys: {len(missing_keys)} (expected: classifier head)")
            print(f"  Unexpected keys: {len(unexpected_keys)}")
            print("✓ Successfully loaded pretrained weights from .npz file")
        
        return True
        
    except Exception as e:
        if is_main_process():
            print(f"❌ Error loading .npz weights: {e}")
        return False


# ============================================================================
# DDP Setup and Utilities
# ============================================================================
def setup_ddp(rank, world_size):
    """Initialize DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP"""
    if dist.is_initialized():
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
# Model Definitions
# ============================================================================
class ViT_L_16_CIFAR(nn.Module):
    """ViT-L/16 for CIFAR-100 with optional pretrained weights"""
    def __init__(self, num_classes=100, pretrained=True, drop_path_rate=0.1, npz_path=None):
        super().__init__()
        
        if pretrained and npz_path is not None:
            # 优先从本地 .npz 文件加载预训练权重
            if is_main_process():
                print(f"Attempting to load from local .npz file: {npz_path}")
            
            if os.path.exists(npz_path):
                # 创建未预训练的模型
                self.model = timm.create_model(
                    'vit_large_patch16_224',
                    pretrained=False,
                    num_classes=num_classes,
                    drop_path_rate=drop_path_rate,
                    img_size=224
                )
                
                # 从 .npz 文件加载权重
                success = load_npz_weights(self.model, npz_path, num_classes)
                
                if success:
                    if is_main_process():
                        print(f"✓ Loaded ViT-L/16 with local pretrained weights from {npz_path}")
                else:
                    if is_main_process():
                        print(f"  Warning: Failed to load from .npz, will use timm pretrained weights")
                    # 回退到 timm 预训练权重
                    self.model = timm.create_model(
                        'vit_large_patch16_224',
                        pretrained=True,
                        num_classes=num_classes,
                        drop_path_rate=drop_path_rate,
                        img_size=224
                    )
                    if is_main_process():
                        print(f"✓ Loaded ViT-L/16 with timm pretrained weights")
            else:
                if is_main_process():
                    print(f"  Warning: .npz file not found at {npz_path}, using timm pretrained weights")
                self.model = timm.create_model(
                    'vit_large_patch16_224',
                    pretrained=True,
                    num_classes=num_classes,
                    drop_path_rate=drop_path_rate,
                    img_size=224
                )
                if is_main_process():
                    print(f"✓ Loaded ViT-L/16 with timm pretrained weights")
        
        elif pretrained:
            # 使用 timm 的在线预训练权重
            try:
                self.model = timm.create_model(
                    'vit_large_patch16_224.augreg_in21k_ft_in1k',
                    pretrained=True,
                    num_classes=num_classes,
                    drop_path_rate=drop_path_rate,
                    img_size=224
                )
                if is_main_process():
                    print(f"✓ Loaded ViT-L/16 with ImageNet-21k pretrained weights")
            except Exception as e:
                if is_main_process():
                    print(f"  Warning: ImageNet-21k weights not available, using ImageNet-1k: {e}")
                self.model = timm.create_model(
                    'vit_large_patch16_224',
                    pretrained=True,
                    num_classes=num_classes,
                    drop_path_rate=drop_path_rate,
                    img_size=224
                )
                if is_main_process():
                    print(f"✓ Loaded ViT-L/16 with ImageNet-1k pretrained weights")
        else:
            # 从头训练
            self.model = timm.create_model(
                'vit_large_patch16_224',
                pretrained=False,
                num_classes=num_classes,
                drop_path_rate=drop_path_rate,
                img_size=224
            )
            if is_main_process():
                print(f"✓ Created ViT-L/16 from scratch")
        
        if is_main_process():
            print(f"  Model params: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")

    def forward(self, x):
        return self.model(x)


class CoupledModel(nn.Module):
    def __init__(self, width=224, num_classes=100, kb=1.0, in_channels=3, pretrained=True, drop_path_rate=0.1, rank=0, npz_path=None):
        super().__init__()
        self.C  = num_classes
        self.kb = kb
        self.rank = rank
        self.device_E = torch.device(f"cuda:{rank}")
        self.device_S = torch.device(f"cuda:{rank}")

        # 仅两张子网：E_net 和 S_net；每个输出 [B, C]
        # 使用 ViT-L/16 替代 ResNet18
        self.E_net = ViT_L_16_CIFAR(num_classes=num_classes, pretrained=pretrained, 
                                     drop_path_rate=drop_path_rate, npz_path=npz_path).to(self.device_E)
        self.S_net = ViT_L_16_CIFAR(num_classes=num_classes, pretrained=pretrained, 
                                     drop_path_rate=drop_path_rate, npz_path=npz_path).to(self.device_S)

    def forward_image(self, x_img):
        """
        前向提取 E(x) 与 S(x)，并分配到各自 GPU
        """
        x_E = x_img.to(self.device_E, non_blocking=True)

        with torch.cuda.device(self.device_E):
            E_mat = self.E_net(x_E)

        # 确保输入在 S_net 所在卡
        x_S = x_img.to(self.device_S, non_blocking=True)
        with torch.cuda.device(self.device_S):
            S_mat = self.S_net(x_S)

        # 回到 device_E 上统一
        E_mat = E_mat.to(self.device_E, non_blocking=True).float()
        S_mat = S_mat.to(self.device_E, non_blocking=True).float()

        sub_outs = torch.stack([E_mat, S_mat], dim=1)
        return E_mat, S_mat, sub_outs
    
    def _normalize_T(self, T, B, device):
        """
        将多种形状的 T 归一到 [B, M, 1] 以便广播：
          允许 [B,1], [M,1], [B,M], [M], 标量
        """
        if not torch.is_tensor(T):
            T = torch.tensor(T, dtype=torch.float32, device=device)

        T = T.to(device)
        if T.dim() == 0:             # 标量 -> [1,1,1]
            T = T.view(1, 1, 1)
        elif T.dim() == 1:           # [M] -> [1,M,1]
            T = T.view(1, -1, 1)
        elif T.dim() == 2:
            if T.size(1) == 1:       # [B,1] 或 [M,1]
                if T.size(0) == B:   # [B,1] -> [B,1,1]
                    T = T.view(B, 1, 1)
                else:                # [M,1] -> [1,M,1]
                    T = T.view(1, -1, 1)
            else:                    # [B,M] -> [B,M,1]
                T = T.unsqueeze(-1)
        # 若本来是 [B,M,1] 则直接返回
        return T
    
    def forward(self, x_img, T):
        """
        x_img:  [B, Cimg, H, W]
        T:      标量或张量，可为 [B,1]/[M,1]/[B,M]/[M]，会自动广播
        返回:
          class_probs: [B, C]            概率（若 T 有 M>1，默认对 T 均匀边缘化）
          scores:      [B, C]            未归一化得分（同上）
          sub_outs:    [B, 2, C]         E/S 调试输出
        """
        device = x_img.device
        eps = 1e-9
        B = x_img.size(0)

        # 一次性取出 E(x), S(x)
        E_mat, S_mat, sub_outs = self.forward_image(x_img)  # [B,C], [B,C], [B,2,C]
        S_pos = F.softplus(S_mat)
        
        # 归一化 T -> [B, M, 1]
        T = self._normalize_T(T, B, device)                 # [B,M,1] 或 [1,M,1]
        # 广播 E/S 到 [B, M, C]
        E_b = E_mat.unsqueeze(1)                            # [B,1,C]
        S_b = S_pos.unsqueeze(1)                            # [B,1,C]

        # 按能量/打分公式构建 scores（逐温度逐类）: [B, M, C]
        scores_bmc = - (E_b - T * S_b) / (self.kb * (T + eps)) - (S_b / (100.0 * self.kb))**2
        probs_bmc  = F.softmax(scores_bmc, dim=2)           # [B, M, C]

        # 若 T 是每样本单值（M=1），压回 [B,C]；否则默认对 T 均匀边缘化后返回 [B,C]
        if scores_bmc.size(1) == 1:
            scores = scores_bmc.squeeze(1)                  # [B,C]
            probs  = probs_bmc.squeeze(1)                   # [B,C]
        else:
            scores = scores_bmc.mean(dim=1)                 # [B,C]
            probs  = probs_bmc.mean(dim=1)                  # [B,C]

        return probs, scores, sub_outs


class LearnableTSet(nn.Module):
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
def em_train_step_optimized_T(model, x, y_onehot, T_module, optimizer, scheduler, scaler, grad_clip=1.0, grad_accum_steps=1):
    """
    E-step + Worst-T M-step: 针对每个样本选择 CE 最大的 T_m 用于反向传播
    输入：
      x:        [N, 3, 224, 224]
      y_onehot: [N, 10]
      T_module: 可学习温度模块
      grad_clip: 梯度裁剪阈值
      grad_accum_steps: 梯度累积步数
    """
    model.train()
    T_grid = T_module()                           # [M]
    N, C = y_onehot.shape
    M = T_grid.size(0)
    device_E = x.device

    with autocast('cuda', dtype=torch.bfloat16):
        # 一次前向图像提取
        E_mat, S_mat, _ = model.module.forward_image(x) if hasattr(model, 'module') else model.forward_image(x)
        S_pos = F.softplus(S_mat)
        
        # 构建 score 矩阵 [N, M, C]
        T_norm = model.module._normalize_T(T_grid, B=N, device=device_E) if hasattr(model, 'module') else model._normalize_T(T_grid, B=N, device=device_E)
        E_b = E_mat.unsqueeze(1)                # [N,1,C]
        S_b = S_pos.unsqueeze(1)                # [N,1,C]
        eps = 1e-9
        kb = model.module.kb if hasattr(model, 'module') else model.kb
        scores_bmc = - (E_b - T_norm * S_b) / (kb * (T_norm + eps)) - (S_b / (100.0 * kb)) ** 2
        log_probs_bmc = F.log_softmax(scores_bmc, dim=2)     # [N, M, C]

        # CE: [N, M]
        ce_bm = -(y_onehot.unsqueeze(1) * log_probs_bmc).sum(dim=2)

        # E-step: compute posterior q(T|x,y)
        qT = torch.softmax(-ce_bm.detach(), dim=1)  # [N, M], detach 不传梯度到 CE

        # M-step: 加权交叉熵
        lambda_sharp = 5.0
        qT = torch.softmax(-lambda_sharp * ce_bm, dim=1)
        loss = (qT * ce_bm).sum() / N # worst-T
        
        # Scale loss for gradient accumulation
        loss = loss / grad_accum_steps

    # backward
    scaler.scale(loss).backward()

    # 返回 loss 以及用于分析的 worst-T 分布（可选）
    return loss.item() * grad_accum_steps, qT.detach()


@torch.no_grad()
def evaluate_accuracy_posterior_labeled(model, loader, device, T_module):
    """
    q(T|x,y) ∝ exp(-CZ(p(y|x,T), y))，
    q(T|x,y) 对 p(y|x,T) 加权边缘化，取 argmax 评估准确率。
    """
    model.eval()
    correct = total = 0

    # 获取当前温度点 T_grid（调用 module）
    T_grid = T_module().detach().to(device)  # [M,1]
    M = T_grid.size(0)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        N, Cimg, H, W = x.shape

        # 展开 (x, T)
        x_rep = x.unsqueeze(1).repeat(1, M, 1, 1, 1).reshape(N*M, Cimg, H, W)
        T_rep = T_grid.view(1, M, 1).expand(N, M, 1).reshape(N*M, 1)

        # 前向 (使用混合精度加速)
        with autocast('cuda', dtype=torch.bfloat16):
            if hasattr(model, 'module'):
                probs, scores, *_ = model.module(x_rep, T_rep)
            else:
                probs, scores, *_ = model(x_rep, T_rep)
                
        Ccls = probs.size(1)
        probs_nm  = probs.view(N, M, Ccls)              # [N, M, C]
        scores_nm = scores.view(N, M, Ccls)             # [N, M, C]

        # 真实标签的逐温度交叉熵
        log_probs_nm = F.log_softmax(scores_nm, dim=2)  # [N, M, C]
        y_onehot = F.one_hot(y, num_classes=Ccls).float()          # [N, C]
        ce_mat = -(y_onehot.unsqueeze(1) * log_probs_nm).sum(dim=2)  # [N, M]

        # E-step: q(T|x,y)
        qT = torch.softmax(-ce_mat, dim=1)              # [N, M]

        # 用后验加权边缘化得到 p(y|x)
        probs_marg_q = (qT.unsqueeze(-1) * probs_nm).sum(dim=1)  # [N, C]

        # 预测与统计
        pred = probs_marg_q.argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += N

    return correct / total


@torch.no_grad()
def posterior_T_labeled_all(model, loader, T_module, device):
    """计算整个数据集的后验概率矩阵"""
    model.eval()
    all_qT = []
    all_Tmap = []
    all_idx = []

    T_grid = T_module().detach().to(device)
    M = T_grid.size(0)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        N, Cimg, H, W = x.shape

        # 展开 (x, T)
        x_rep = x.unsqueeze(1).repeat(1, M, 1, 1, 1).reshape(N*M, Cimg, H, W)
        T_rep = T_grid.view(1, M, 1).expand(N, M, 1).reshape(N*M, 1)
        y_rep = y.unsqueeze(1).repeat(1, M).reshape(N*M).long()

        # 模型前向 (使用混合精度)
        with autocast('cuda', dtype=torch.bfloat16):
            if hasattr(model, 'module'):
                probs, scores, *_ = model.module(x_rep, T_rep)
            else:
                probs, scores, *_ = model(x_rep, T_rep)
                
        log_probs = F.log_softmax(scores, dim=1)

        # CE loss
        ce_vec = F.nll_loss(log_probs, y_rep, reduction="none")
        ce_mat = ce_vec.view(N, M)

        # 后验 q(T|x,y)
        qT = torch.softmax(-ce_mat, dim=1)

        # MAP 温度
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
    
    # If test_epochs not provided, assume test_accuracies correspond to epochs_range
    if test_epochs is None:
        # Generate test_epochs based on the number of test_accuracies
        if len(test_accuracies) == len(train_losses):
            # Test accuracy recorded every epoch
            test_epochs = epochs_range
        else:
            # Try to infer the recording pattern
            # Assume uniform distribution over the training period
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
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    
    # Performance optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device = torch.device(f"cuda:{rank}")
    
    if is_main_process():
        print(f"\n{'='*80}")
        print(f"ViT-L/16 CoupledModel DDP Training")
        print(f"{'='*80}")
        print(f"Using {world_size} GPU(s) with DDP")
        for i in range(world_size):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"\nConfiguration:")
        print(f"  Model: ViT-L/16 (Large)")
        print(f"  Input Resolution: 224x224")
        if args.pretrained:
            if args.npz_path:
                print(f"  Pretrained: Local .npz file ({args.npz_path})")
            else:
                print(f"  Pretrained: ImageNet (timm)")
        else:
            print(f"  Pretrained: None (train from scratch)")
        print(f"  Batch Size per GPU: {args.batch_size}")
        print(f"  Gradient Accumulation Steps: {args.grad_accum_steps}")
        print(f"  Global Batch Size: {args.batch_size * world_size * args.grad_accum_steps}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Learning Rate (E/S): {args.lr}")
        print(f"  Learning Rate (T): {args.lr_t}")
        print(f"  Weight Decay: {args.weight_decay}")
        print(f"  Warmup Epochs: {args.warmup_epochs}")
        print(f"  Gradient Clip: {args.grad_clip}")
        print(f"  Optimizer: AdamW")
        print(f"  Scheduler: Warmup + Cosine Decay")
        print(f"  Mixed Precision: bfloat16")
        print(f"  K (Learnable T): {args.K}")
        print(f"  T Range: [{args.T_min}, {args.T_max}]")
        print(f"  Drop Path Rate: {args.drop_path_rate}")
        print(f"{'='*80}\n")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # Load datasets
    if is_main_process():
        print("Loading datasets...")
    
    train_dataset = CIFAR100(root=args.data_dir, train=True, download=True, transform=train_transform)
    test_dataset = CIFAR100(root=args.data_dir, train=False, download=True, transform=eval_transform)
    
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
        print("\nInitializing models...")
    
    model = CoupledModel(
        width=224, 
        num_classes=100, 
        kb=args.kb, 
        in_channels=3, 
        pretrained=args.pretrained, 
        drop_path_rate=args.drop_path_rate,
        rank=rank,
        npz_path=args.npz_path
    ).to(device)
    
    T_module = LearnableTSet(K=args.K, T_min=args.T_min, T_max=args.T_max).to(device)
    
    # Wrap with DDP (only wrap model, not T_module as it's small and doesn't need DDP)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    # T_module is not wrapped with DDP to avoid issues with forward() that takes no arguments
    
    # Optimizer and scheduler
    params = [
        {"params": model.module.E_net.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": model.module.S_net.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": T_module.parameters(), "lr": args.lr_t, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, betas=(0.9, 0.999))
    
    # Learning rate scheduler with warmup
    total_steps = args.epochs * len(train_loader) // args.grad_accum_steps
    warmup_steps = args.warmup_epochs * len(train_loader) // args.grad_accum_steps
    
    def get_lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda)
    scaler = GradScaler('cuda', enabled=True)
    
    if is_main_process():
        print("✓ Model and optimizer initialized")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params / 1e6:.1f}M")
        print(f"Starting training...\n")
    
    # Training loop
    train_losses = []
    test_accuracies = []
    test_epochs_list = []  # Track which epochs test accuracy was recorded
    T_records = []
    
    training_start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        train_sampler.set_epoch(epoch)
        
        # Synchronize T_module parameters at the start of each epoch
        if dist.is_initialized():
            for param in T_module.parameters():
                dist.broadcast(param.data, src=0)
        
        total_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y_onehot = F.one_hot(y, num_classes=100).float().to(device, non_blocking=True)
            
            # EM + AMP 优化 + Learnable T + Gradient Accumulation
            loss, qT = em_train_step_optimized_T(
                model, x, y_onehot, T_module, optimizer, scheduler, scaler,
                grad_clip=args.grad_clip, grad_accum_steps=args.grad_accum_steps
            )
            total_loss += loss
            
            # Gradient accumulation
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                # Synchronize T_module gradients across all GPUs (since it's not wrapped with DDP)
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
                
                # Update learning rate
                scheduler.step()
        
        # 记录
        avg_loss = total_loss / len(train_loader)
        
        # Synchronize loss across all processes
        if dist.is_initialized():
            loss_tensor = torch.tensor([avg_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        # 测试集评估
        with torch.no_grad():
            T_eval_grid = T_module.module.forward() if hasattr(T_module, 'module') else T_module()
            current_T = T_eval_grid.detach().cpu().numpy().flatten()
        
        # Record on main process
        if is_main_process():
            train_losses.append(avg_loss)
            T_records.append(current_T.copy())
        
        # 每 print_freq epoch 打印测试准确率
        if epoch % args.print_freq == 0 or epoch == args.epochs - 1:
            test_acc = evaluate_accuracy_posterior_labeled(model, test_loader, device, T_module.module if hasattr(T_module, 'module') else T_module)
            
            # Synchronize test accuracy
            if dist.is_initialized():
                test_acc_tensor = torch.tensor([test_acc], device=device)
                dist.all_reduce(test_acc_tensor, op=dist.ReduceOp.AVG)
                test_acc = test_acc_tensor.item()
            
            if is_main_process():
                test_accuracies.append(test_acc)
                test_epochs_list.append(epoch + 1)  # Record which epoch this accuracy is from (1-indexed)
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch:3d}: Loss = {avg_loss:.6f}, Test Acc = {test_acc:.6f}, "
                      f"Time = {epoch_time:.2f}s, T = {current_T}")
    
    if is_main_process():
        total_time = time.time() - training_start_time
        print(f"\nTraining complete! Total time: {total_time/3600:.2f}h")
        
        # 计算后验概率矩阵
        print("\nComputing posterior T distributions...")
        qT_all, Tmap_all, idx_all = posterior_T_labeled_all(model, test_loader, T_module.module if hasattr(T_module, 'module') else T_module, device)
        
        # 计算频率
        max_idx = torch.argmax(qT_all, dim=1)
        qT_onehot = F.one_hot(max_idx, num_classes=qT_all.shape[1]).float()
        counts = qT_onehot.sum(dim=0)
        freqs = counts / qT_onehot.shape[0]
        
        print(f"Temperature distribution:")
        print(f"  Counts: {counts.numpy()}")
        print(f"  Frequencies: {freqs.numpy()}")
        
        # 保存结果
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving results to {output_dir}...")
        np.savetxt(os.path.join(output_dir, "train_losses.txt"), np.array(train_losses), fmt="%.10f")
        np.savetxt(os.path.join(output_dir, "test_accuracies.txt"), np.array(test_accuracies), fmt="%.10f")
        np.savetxt(os.path.join(output_dir, "test_epochs.txt"), np.array(test_epochs_list), fmt="%d")
        np.savetxt(os.path.join(output_dir, "T_records.txt"), np.array(T_records), fmt="%.10f")
        np.savetxt(os.path.join(output_dir, "freqs.txt"), freqs.numpy(), fmt="%.10f")
        np.savetxt(os.path.join(output_dir, "qT_all.txt"), qT_all.numpy(), fmt="%.4f")
        
        # 绘制训练曲线
        plot_training_curves(train_losses, test_accuracies, T_records, 
                           os.path.join(output_dir, "training_curves.png"),
                           test_epochs=test_epochs_list)
        
        # 保存模型
        torch.save({
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'T_module_state_dict': T_module.module.state_dict() if hasattr(T_module, 'module') else T_module.state_dict(),
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
def main():
    parser = argparse.ArgumentParser(description='ViT-L/16 CoupledModel Training with DDP')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='./data', help='path to dataset')
    parser.add_argument('--output-dir', type=str, default='./output_ddp', help='path to save outputs')
    
    # Model arguments
    parser.add_argument('--pretrained', action='store_true', default=True, help='use pretrained weights')
    parser.add_argument('--npz-path', type=str, default='./pretrained_models/ViT-L_16.npz', 
                        help='path to local .npz pretrained weights file')
    parser.add_argument('--drop-path-rate', type=float, default=0.1, help='drop path rate')
    parser.add_argument('--kb', type=float, default=1.0, help='Boltzmann constant')
    
    # Temperature arguments
    parser.add_argument('--K', type=int, default=4, help='number of learnable temperatures')
    parser.add_argument('--T-min', type=float, default=0.1, help='minimum temperature')
    parser.add_argument('--T-max', type=float, default=10.0, help='maximum temperature')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=101, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size per GPU')
    parser.add_argument('--grad-accum-steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='gradient clipping threshold')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for E/S nets')
    parser.add_argument('--lr-t', type=float, default=1e-3, help='learning rate for T module')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='number of warmup epochs')
    parser.add_argument('--num-workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency')
    parser.add_argument('--seed', type=int, default=100, help='random seed')
    
    args = parser.parse_args()
    
    # Get number of GPUs
    world_size = torch.cuda.device_count()
    
    if world_size == 0:
        print("No CUDA devices available!")
        return
    
    print(f"Found {world_size} CUDA devices")
    
    # Launch DDP training
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()

