#!/bin/bash
#SBATCH --job-name=vit_h14_cifar100
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# Print job information
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "=================================="

# Activate conda environment
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate NN

# Set environment variables for DDP
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Performance optimizations
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3

# Change to working directory
cd /home/spw5793/work/ZENN/Cifar100/ViT_L_16

# Print GPU information
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Run DDP optimized training with ImageNet-21k pretrained weights
# Key optimizations:
# - DistributedDataParallel (DDP) for 4 GPUs
# - bfloat16 mixed precision (A100 optimized)
# - channels_last memory format
# - Gradient checkpointing (saves memory)
# - TF32 enabled (A100 optimized)
# - Global batch size = 512 (64 per GPU × 4 GPUs × 2 accum steps)
# - Optimized data loading (num_workers=4 per GPU)
# - ImageNet-21k pretrained weights via timm

python train_ddp_optimized.py \
    --data-dir ./data \
    --output-dir ./output_ddp_optimized_in21k \
    --epochs 50 \
    --batch-size 64 \
    --grad-accum-steps 2 \
    --grad-clip 1.0 \
    --lr 1e-4 \
    --weight-decay 0.1 \
    --warmup-epochs 5 \
    --mixup-alpha 0.2 \
    --cutmix-alpha 0.1 \
    --mixup-prob 0.5 \
    --cutmix-prob 0.5 \
    --label-smoothing 0.1 \
    --ema-decay 0.9999 \
    --drop-path-rate 0.1 \
    --num-workers 4 \
    --print-freq 1 \
    --seed 100 \
    --pretrained \
    --use-checkpoint \
    --use-amp

echo ""
echo "=================================="
echo "End time: $(date)"
echo "=================================="

# Print final GPU memory usage
echo ""
echo "Final GPU Memory Usage:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

