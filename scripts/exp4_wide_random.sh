#!/bin/bash
#SBATCH --job-name=ev_exp4_wide
#SBATCH --output=/home/a5l/shuqing.a5l/OpenMythos/scripts/logs/exp4_%j.log
#SBATCH --error=/home/a5l/shuqing.a5l/OpenMythos/scripts/logs/exp4_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Exp 4: random_width samples from U[1, max_streams] instead of U[1, n_streams]
# This lets the model see stream counts beyond the training default during
# training, which should improve width extrapolation to 8 streams.
# ============================================

set -e

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/openmythos/bin:$PATH"
export CONDA_DEFAULT_ENV=openmythos
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/home/a5l/shuqing.a5l/OpenMythos"
LOG_DIR="$PROJECT_ROOT/scripts/logs"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "Exp 4: random_width ~ U[1, max_streams=8]"
echo "Job: ${SLURM_JOB_NAME}  ID: ${SLURM_JOB_ID}"
echo "Start: $(date)"
echo "=========================================="

python tests/small_benchmark.py \
    --steps 5000 --batch-size 64 --seq-len 512 --device cuda \
    --log-every 100 --eval-every 500 \
    --n-streams 4 --max-streams 8 \
    --random-depth --random-width --contraction-weight 0.01 \
    --width-sweep 1,2,4,8 --depth-sweep 1,2,4,8,16 \
    --save-log "$LOG_DIR/exp4_wide_random_${SLURM_JOB_ID}.json"

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
