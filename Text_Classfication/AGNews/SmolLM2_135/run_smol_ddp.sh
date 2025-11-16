#!/bin/bash
#SBATCH --job-name=smollm2_full_finetune
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# Print job information
echo "=================================="
echo "SmolLM2-135M Full Fine-tuning on AG News with DDP"
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
cd /home/spw5793/work/ZENN/AgNews/SmolLM2_all_parameter

# Print GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Print configuration
echo "Training Configuration:"
echo "  Model: SmolLM2-135M (from Hugging Face Hub)"
echo "  Pretrained: HuggingFaceTB/SmolLM2-135M"
echo "  Fine-tuning: Full Fine-tuning (all parameters)"
echo "  Optimizer: Adam weight decay optimizer (AdamW)"
echo "  Learning Rate: 5e-5"
echo "  Weight Decay: 0.01"
echo "  Learning rate decay: Cosine with warmup"
echo "  Dataset: AG News (4 classes)"
echo "  GPUs: 4 (CUDA:4,5,6,7)"
echo "  Batch Size per GPU: 64"
echo "  Gradient Accumulation Steps: 2"
echo "  Global Batch Size: 512 (64 x 4 x 2)"
echo "  Epochs: 6"
echo "  Warmup Epochs: 1"
echo "  Max Sequence Length: 512"
echo "  Mixed Precision: bfloat16"
echo "  Gradient Checkpointing: Enabled"
echo "  EMA: Enabled"
echo ""

# Run DDP training (Full Fine-tuning)
python train_smol_ddp.py \
    --output-dir ./output_smol_ddp \
    --epochs 10 \
    --batch-size 32 \
    --grad-accum-steps 2 \
    --grad-clip 1.0 \
    --lr 5e-5 \
    --weight-decay 0.01 \
    --warmup-epochs 1 \
    --max-length 512 \
    --num-workers 4 \
    --print-freq 1 \
    --seed 42 \
    --use-amp \
    --use-checkpoint

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
echo "Output files saved to: ./output_smol_ddp/"
echo "  - train_losses.txt"
echo "  - test_accuracies.txt"
echo "  - learning_rates.txt"
echo "  - epoch_times.txt"
echo "  - training_curves.png"


