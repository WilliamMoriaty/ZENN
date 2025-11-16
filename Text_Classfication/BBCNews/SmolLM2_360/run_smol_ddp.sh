#!/bin/bash
#SBATCH --job-name=smollm2_360_lora_bbcnews
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# ============================================================================
# SmolLM2-360M LoRA Fine-tuning Configuration
# ============================================================================
# This script uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
# Requirements:
#   - transformers: Hugging Face 模型库
#   - peft: LoRA 适配器
#   - datasets: BBC News 数据集
#
# To install dependencies:
#   pip install transformers peft datasets accelerate
# ============================================================================

# Print job information
echo "=================================="
echo "SmolLM2-360M with LoRA on BBC News with DDP"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "=================================="
echo ""

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

# Workspace config
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_CUDNN_V8_API_ENABLED=1

# Change to working directory
cd /home/spw5793/work/ZENN/BBCNews/SmolLM2_360

# Print GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Print configuration
echo "Training Configuration (Optimized for BBC News Small Dataset):"
echo "  Model: SmolLM2-360M (360M parameters)"
echo "  Fine-tuning: LoRA (Low-Rank Adaptation, Parameter-Efficient)"
echo "  LoRA Rank: 8"
echo "  LoRA Alpha: 16"
echo "  LoRA Target Modules: q_proj, v_proj (Llama/SmolLM2 attention)"
echo "  Optimizer: Adam weight decay optimizer (AdamW)"
echo "  Learning Rate: 5e-5 (optimized for LoRA)"
echo "  Weight Decay: 0.01"
echo "  Learning rate decay: Cosine with warmup"
echo "  Dataset: BBC News (5 classes, 1,225 train samples - SMALL DATASET!)"
echo "  GPUs: 4 (CUDA:0,1,2,3)"
echo "  Batch Size per GPU: 4 (reduced for small dataset)"
echo "  Gradient Accumulation Steps: 1"
echo "  Global Batch Size: 16 (4 x 4 x 1)"
echo "  Steps per Epoch: ~77 (much better for learning!)"
echo "  Epochs: 30"
echo "  Warmup Epochs: 1"
echo "  Max Sequence Length: 512"
echo "  Mixed Precision: bfloat16"
echo "  Gradient Checkpointing: Disabled (incompatible with LoRA + DDP)"
echo "  EMA: Disabled (not needed with LoRA)"
echo "  Target Memory: ~15GB per GPU"
echo "  Expected Speed: ~20-30s/epoch (small dataset, 360M model)"
echo ""

# Run DDP training with LoRA (Optimized for Small Dataset)
python train_smol_ddp.py \
    --output-dir ./output_smollm2_360_lora_bbc \
    --epochs 30 \
    --batch-size 4 \
    --grad-accum-steps 1 \
    --grad-clip 1.0 \
    --lr 5e-5 \
    --weight-decay 0.01 \
    --warmup-epochs 1 \
    --max-length 512 \
    --num-workers 4 \
    --print-freq 1 \
    --seed 42 \
    --use-amp \
    --use-lora \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-dropout 0.1 \
    --lora-target-modules q_proj,v_proj

echo ""
echo "=================================="
echo "End time: $(date)"
echo "=================================="

# Print final GPU memory usage
echo ""
echo "Final GPU Memory Usage:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# Print output files location
echo ""
echo "Output files saved to: ./output_smollm2_360_lora_bbc/"
echo "  - train_losses.txt"
echo "  - test_accuracies.txt"
echo "  - learning_rates.txt"
echo "  - epoch_times.txt"
echo "  - training_curves.png"
echo ""
echo "BBC News Optimized LoRA Configuration Summary:"
echo "  ✓ Model: SmolLM2-360M (Llama architecture)"
echo "  ✓ Dataset: BBC News (5 classes, 1,225 samples)"
echo "  ✓ Fine-tuning method: LoRA (Parameter-Efficient)"
echo "  ✓ LoRA Rank: 8, Target: q_proj, v_proj"
echo "  ✓ Trainable parameters: ~0.8M (<0.23% of total)"
echo "  ✓ Batch size: 4/GPU (optimized for small dataset)"
echo "  ✓ Gradient accumulation: 1 step"
echo "  ✓ Sequence length: 512"
echo "  ✓ Global batch size: 16 (not 512!)"
echo "  ✓ Steps per epoch: ~77 (was only 2-3 before!)"
echo "  ✓ Memory per GPU: ~15GB"
echo "  ✓ Expected speed: ~20-30s/epoch"
echo "  ✓ GPUs: 4"
echo "  ✓ KEY FIX: Reduced batch size for small dataset learning!"


