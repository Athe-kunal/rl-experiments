#!/usr/bin/env bash
set -euo pipefail

# Example:
#   bash examples/train/algorithms/dapo/run_dapo_openr1_fsdp_skyrl_train_docker.sh
#
# This script:
#  1) starts a SkyRL docker container with GPUs
#  2) prepares open-r1 DAPO math dataset inside the container
#  3) launches FSDP DAPO training via skyrl-train

IMAGE=${IMAGE:-"ghcr.io/skyrl-team/skyrl-train:latest"}
CONTAINER_NAME=${CONTAINER_NAME:-"skyrl-dapo-openr1"}
HOST_REPO_ROOT=${HOST_REPO_ROOT:-"$(cd "$(dirname "$0")/../../../.." && pwd)"}
CONTAINER_REPO_ROOT=${CONTAINER_REPO_ROOT:-"/workspace/SkyRL"}
DATA_DIR=${DATA_DIR:-"${CONTAINER_REPO_ROOT}/data/dapo"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/dapo-math-17k-openr1-processed.parquet"}
TEST_FILE=${TEST_FILE:-"${DATA_DIR}/aime-2024.parquet"}

MODEL=${MODEL:-"Qwen/Qwen3-1.7B"}
WANDB_PROJECT=${WANDB_PROJECT:-""}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}

if ! docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker run -d --gpus all --ipc=host --shm-size=128g \
    --name "${CONTAINER_NAME}" \
    -v "${HOST_REPO_ROOT}:${CONTAINER_REPO_ROOT}" \
    -w "${CONTAINER_REPO_ROOT}" \
    "${IMAGE}" sleep infinity
fi

# prepare data

docker exec -e DATA_DIR="${DATA_DIR}" "${CONTAINER_NAME}" bash -lc '
set -euo pipefail
uv run --isolated --extra fsdp -m examples.train.algorithms.dapo.prepare_openr1_dapo_math_data --data-dir "${DATA_DIR}"
'

# train

docker exec \
  -e WANDB_PROJECT="${WANDB_PROJECT}" \
  -e NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE}" \
  -e MODEL="${MODEL}" \
  -e TRAIN_FILE="${TRAIN_FILE}" \
  -e TEST_FILE="${TEST_FILE}" \
  "${CONTAINER_NAME}" bash -lc '
set -euo pipefail
uv run --isolated --extra fsdp -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="[\"${TRAIN_FILE}\"]" \
  data.val_data="[\"${TEST_FILE}\"]" \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.fsdp_size="${NUM_GPUS_PER_NODE}" \
  trainer.policy.model.path="${MODEL}" \
  trainer.logger.backends="[wandb]" \
  trainer.logger.project="${WANDB_PROJECT}"
'
