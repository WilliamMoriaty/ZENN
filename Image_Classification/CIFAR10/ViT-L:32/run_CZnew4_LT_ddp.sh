#!/bin/bash
#SBATCH --job-name=vit_l32_coupled_cifar10
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
echo "ViT-L/32 CoupledModel DDP Training"
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
export CUDA_VISIBLE_DEVICES=4,5,6,7
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
cd /home/spw5793/work/ZENN/Cifar10/ViT_L_32

# Print GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Print configuration
echo "Training Configuration:"
echo "  Model: ViT-L/32 CoupledModel (Large, patch size 32)"
echo "  Dataset: CIFAR-10"
echo "  Pretrained Weights: ./pretrained_models/ViT-L_32.npz"
echo "  GPUs: 4 (CUDA:0,1,2,3)"
echo "  Batch Size per GPU: 64"
echo "  Gradient Accumulation Steps: 2"
echo "  Global Batch Size: 512 (64 x 4 x 2)"
echo "  Epochs: 50"
echo "  Learning Rate (E/S): 1e-4"
echo "  Learning Rate (T): 1e-3"
echo "  Weight Decay: 0.1"
echo "  Warmup Epochs: 5"
echo "  Gradient Clip: 1.0"
echo "  Optimizer: AdamW"
echo "  Mixed Precision: bfloat16"
echo "  Temperature Learning: K=2, Range=[0.1, 10.0]"
echo ""

# Run DDP training
# Following train_ddp_optimized.py configuration:
# - batch-size 64 (per GPU)
# - grad-accum-steps 2 (effective global batch = 64 x 4 x 2 = 512)
# - epochs 50
# - weight-decay 0.1
# - warmup-epochs 5
# - grad-clip 1.0
python CZnew4_LT_image_ddp.py \
    --data-dir ./data \
    --output-dir ./output_coupled_ddp_K2 \
    --epochs 50 \
    --batch-size 64 \
    --grad-accum-steps 2 \
    --grad-clip 1.0 \
    --lr 1e-4 \
    --lr-t 1e-3 \
    --weight-decay 0.1 \
    --warmup-epochs 5 \
    --K 2 \
    --T-min 0.1 \
    --T-max 10.0 \
    --drop-path-rate 0.1 \
    --num-workers 4 \
    --print-freq 1 \
    --seed 100 \
    --pretrained \
    --npz-path ./pretrained_models/ViT-L_32.npz

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
echo "Output files saved to: ./output_coupled_ddp_K2/"
echo "  - train_losses.txt"
echo "  - test_accuracies.txt"
echo "  - T_records.txt"
echo "  - freqs.txt"
echo "  - qT_all.txt"
echo "  - training_curves.png"
echo "  - checkpoint_final.pth"

