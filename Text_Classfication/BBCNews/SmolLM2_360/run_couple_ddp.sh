#!/bin/bash
#SBATCH --job-name=smollm2_360_coupled_bbcnews
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# ============================================================================
# SmolLM2-360M CoupledModel Training with Learnable Temperature
# ============================================================================
# This script trains a coupled model with:
#   - Two sub-networks: E_net and S_net
#   - Learnable temperature module
#   - EM optimization with posterior evaluation
#   - Optional LoRA for parameter-efficient fine-tuning
# 
# Requirements:
#   - transformers: Hugging Face 模型库
#   - peft: LoRA 适配器 (optional)
#   - datasets: BBC News 数据集
#
# To install dependencies:
#   pip install transformers peft datasets accelerate
# ============================================================================

# Print job information
echo "=================================="
echo "SmolLM2-360M CoupledModel Training on BBC News"
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
cd /home/spw5793/work/ZENN/BBCNews/SmolLM2_360

# Print GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Print configuration
echo "Training Configuration (CoupledModel with Learnable Temperature):"
echo "  Architecture: CoupledModel (E_net + S_net)"
echo "  Model: SmolLM2-360M (360M parameters each)"
echo "  Dataset: BBC News (5 classes, 1,225 train samples)"
echo "  Fine-tuning: LoRA (Parameter-Efficient)"
echo "  LoRA Rank: 8"
echo "  LoRA Alpha: 16"
echo "  LoRA Target Modules: q_proj, v_proj"
echo "  Temperature Learning: K=4 learnable temperatures"
echo "  T Range: [0.1, 10.0]"
echo "  Training Method: EM with Worst-T optimization"
echo "  Evaluation: Posterior q(T|x,y) weighted"
echo ""
echo "  Optimizer: AdamW"
echo "  Learning Rate (E/S): 5e-5"
echo "  Learning Rate (T): 1e-3"
echo "  Weight Decay: 0.01"
echo "  Warmup Epochs: 1"
echo "  Gradient Clip: 1.0"
echo ""
echo "  GPUs: 4 (CUDA:0,1,2,3)"
echo "  Batch Size per GPU: 4 (reduced for small dataset)"
echo "  Gradient Accumulation Steps: 1"
echo "  Global Batch Size: 16 (4 x 4 x 1)"
echo "  Steps per Epoch: ~77"
echo "  Epochs: 30"
echo "  Max Sequence Length: 512"
echo "  Mixed Precision: bfloat16"
echo ""
echo "  Expected Memory: ~24GB per GPU (2 x 360M models + LoRA)"
echo "  Expected Speed: ~40-50s/epoch (larger model) seed = 77 (K=1) "
echo ""

# Run DDP training with CoupledModel
python train_couple_ddp.py \
    --output-dir ./output_coupled_360_lora_bbc_K3 \
    --model-name HuggingFaceTB/SmolLM2-360M \
    --hf-token ### \
    --epochs 30 \
    --batch-size 4 \
    --grad-accum-steps 1 \
    --grad-clip 1.0 \
    --lr 5e-5 \
    --lr-t 1e-3 \
    --weight-decay 0.01 \
    --warmup-epochs 1 \
    --max-length 512 \
    --num-workers 4 \
    --print-freq 1 \
    --seed 42 \
    --use-lora \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-dropout 0.1 \
    --lora-target-modules q_proj,v_proj \
    --K 3 \
    --T-min 0.1 \
    --T-max 10.0 \
    --kb 1.0

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
echo "Output files saved to: ./output_coupled_360_lora_bbc_K3/"
echo "  - train_losses.txt"
echo "  - test_accuracies.txt"
echo "  - test_epochs.txt"
echo "  - T_records.txt (temperature evolution)"
echo "  - freqs.txt (temperature distribution)"
echo "  - qT_all.txt (posterior probabilities)"
echo "  - training_curves.png"
echo "  - checkpoint_final.pth"
echo ""
echo "CoupledModel Configuration Summary:"
echo "  ✓ Architecture: CoupledModel with E_net + S_net"
echo "  ✓ Model: SmolLM2-360M (360M × 2 = 720M total params)"
echo "  ✓ Dataset: BBC News (5 classes, 1,225 samples)"
echo "  ✓ Fine-tuning: LoRA (~1.6M trainable params, <0.23% of total)"
echo "  ✓ Temperature: K=1 learnable temps in [0.1, 10.0]"
echo "  ✓ Training: EM with worst-T optimization"
echo "  ✓ Evaluation: Posterior q(T|x,y) weighted accuracy"
echo "  ✓ Batch size: 4/GPU, Global: 16, Steps/epoch: 77"
echo "  ✓ Memory: ~24GB/GPU, Speed: ~40-50s/epoch"
echo "  ✓ Key Innovation: Temperature-aware ensemble learning!"
echo ""
echo "Expected Improvements over Single Model:"
echo "  • Better uncertainty quantification via temperature"
echo "  • Ensemble effect from E_net + S_net"
echo "  • Adaptive temperature per sample via posterior"
echo "  • Potential accuracy boost of 1-3%"
echo ""



