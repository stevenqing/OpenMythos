#!/bin/bash
#SBATCH --job-name=ev_extrap_fix
#SBATCH --output=/home/a5l/shuqing.a5l/OpenMythos/scripts/logs/extrap_fix_%j.log
#SBATCH --error=/home/a5l/shuqing.a5l/OpenMythos/scripts/logs/extrap_fix_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Extrapolation Fix Benchmark
# Description: Compare baseline (no randomization) vs random training
#              + contraction regularization for depth/width extrapolation.
# ============================================

set -e

# ── Environment ──
export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/openmythos/bin:$PATH"
export CONDA_DEFAULT_ENV=openmythos
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# ── Paths ──
PROJECT_ROOT="/home/a5l/shuqing.a5l/OpenMythos"
LOG_DIR="$PROJECT_ROOT/scripts/logs"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Use cached datasets/tokenizers (compute nodes may lack internet)
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "Job: ${SLURM_JOB_NAME}  ID: ${SLURM_JOB_ID}"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "=========================================="

COMMON="--steps 5000 --batch-size 64 --seq-len 512 --device cuda \
    --log-every 100 --eval-every 500"

# ── Exp 1: 4-stream baseline (no randomization) — control ──
echo ""
echo ">>> Exp 1: 4-stream baseline (no randomization)"
echo ""
python tests/small_benchmark.py $COMMON \
    --n-streams 4 --max-streams 8 \
    --width-sweep 1,2,4,8 --depth-sweep 1,2,4,8,16 \
    --save-log "$LOG_DIR/exp1_baseline_${SLURM_JOB_ID}.json"

# ── Exp 2: 4-stream + random-depth + random-width + contraction ──
echo ""
echo ">>> Exp 2: 4-stream + random-depth + random-width + contraction"
echo ""
python tests/small_benchmark.py $COMMON \
    --n-streams 4 --max-streams 8 \
    --random-depth --random-width --contraction-weight 0.01 \
    --width-sweep 1,2,4,8 --depth-sweep 1,2,4,8,16 \
    --save-log "$LOG_DIR/exp2_random_contraction_${SLURM_JOB_ID}.json"

# ── Exp 3: 1-stream + random-depth + contraction — depth only ──
echo ""
echo ">>> Exp 3: 1-stream + random-depth + contraction (depth extrapolation only)"
echo ""
python tests/small_benchmark.py $COMMON \
    --n-streams 1 \
    --random-depth --contraction-weight 0.01 \
    --depth-sweep 1,2,4,8,16 \
    --save-log "$LOG_DIR/exp3_depth_only_${SLURM_JOB_ID}.json"

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
