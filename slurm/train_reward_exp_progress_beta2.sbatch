#!/usr/bin/env bash
#SBATCH --job-name=rw_exp_b2
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch1/haobaizh/rewind_valuemodel/slurm_logs/%x_%j.out
#SBATCH --error=/scratch1/haobaizh/rewind_valuemodel/slurm_logs/%x_%j.err

# Train the ReWiND reward model on normalized exponential progress labels.
# Output checkpoint:
#   /scratch1/haobaizh/rewind_valuemodel/checkpoints/rewind_metaworld_exp_beta2_epoch_*.pth

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/project2/biyik_1165/haobaizh/rewind_valuemodel}"
SCRATCH_DIR="${REWIND_SCRATCH_DIR:-/scratch1/haobaizh/rewind_valuemodel}"
CONDA_ENV="${CONDA_ENV:-${PROJECT_DIR}/conda_envs/rewind}"
WANDB_ENTITY="${WANDB_ENTITY:-rewind-valuemodel}"
WANDB_PROJECT="${WANDB_PROJECT:-rewind-reward-training}"
BETA="${BETA:-2.0}"

mkdir -p \
  "${SCRATCH_DIR}/slurm_logs" \
  "${SCRATCH_DIR}/checkpoints" \
  "${SCRATCH_DIR}/wandb" \
  "${SCRATCH_DIR}/.cache/wandb" \
  "${SCRATCH_DIR}/.config/wandb" \
  "${SCRATCH_DIR}/confusion_matrix_for_paper"

module purge
module load gcc/13.3.0
module load mesa/24.2.3
module load mesa-glu/9.0.2
module load glew/2.2.0
module load glx/1.4

source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export REWIND_SCRATCH_DIR="${SCRATCH_DIR}"
export WANDB_DIR="${SCRATCH_DIR}/wandb"
export WANDB_CACHE_DIR="${SCRATCH_DIR}/.cache/wandb"
export WANDB_CONFIG_DIR="${SCRATCH_DIR}/.config/wandb"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export QT_XCB_GL_INTEGRATION=none
export LD_PRELOAD="$(g++ -print-file-name=libstdc++.so.6)"
export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
GCC_LIBDIR="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="${GCC_LIBDIR}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

cd "${PROJECT_DIR}"

echo "========== Train exponential ReWiND reward model =========="
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "SCRATCH_DIR=${SCRATCH_DIR}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "BETA=${BETA}"
echo "checkpoint_dir=${SCRATCH_DIR}/checkpoints"

srun python -u train_reward.py \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_project "${WANDB_PROJECT}" \
  --h5_folder_path "${PROJECT_DIR}/datasets" \
  --openx_embedding_path "${PROJECT_DIR}/datasets/full_openx_embeddings_v2_train.h5" \
  --rewind \
  --subsample_video \
  --clip_grad \
  --cosine_scheduler \
  --batch_size 1024 \
  --worker 1 \
  --use_exponential_progress \
  --exponential_beta "${BETA}" \
  --checkpoint_dir "${SCRATCH_DIR}/checkpoints" \
  --wandb_dir "${SCRATCH_DIR}/wandb" \
  --confusion_matrix_dir "${SCRATCH_DIR}/confusion_matrix_for_paper"
