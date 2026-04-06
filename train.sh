#!/bin/bash
# Fast multi-GPU ACT training with lerobot
#
# Usage:
#   bash train.sh --dataset-root /dev/shm/sim_pick_vase
#   bash train.sh --dataset-root /dev/shm/sim_pick_vase --num-gpus 4 --batch-size 64
#   bash train.sh --dataset-root /dev/shm/sim_pick_vase --steps 50000 --chunk-size 50

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
DATASET_ROOT=""
REPO_ID="local/sim_pick_vase"
NUM_GPUS=""
BATCH_SIZE=32
NUM_WORKERS=2
STEPS=10000
SAVE_FREQ=5000
LOG_FREQ=100
CHUNK_SIZE=100
N_ACTION_STEPS=50
N_OBS_STEPS=1
SEED=42
EXTRA_ARGS=""

# ── Parse args ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset-root) DATASET_ROOT="$2"; shift 2;;
        --repo-id) REPO_ID="$2"; shift 2;;
        --num-gpus) NUM_GPUS="$2"; shift 2;;
        --batch-size) BATCH_SIZE="$2"; shift 2;;
        --num-workers) NUM_WORKERS="$2"; shift 2;;
        --steps) STEPS="$2"; shift 2;;
        --save-freq) SAVE_FREQ="$2"; shift 2;;
        --log-freq) LOG_FREQ="$2"; shift 2;;
        --chunk-size) CHUNK_SIZE="$2"; shift 2;;
        --n-action-steps) N_ACTION_STEPS="$2"; shift 2;;
        --n-obs-steps) N_OBS_STEPS="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift;;
    esac
done

if [ -z "$DATASET_ROOT" ]; then
    echo "Usage: bash train.sh --dataset-root /dev/shm/sim_pick_vase [options]"
    echo ""
    echo "Options:"
    echo "  --dataset-root PATH    Path to dataset (required)"
    echo "  --repo-id ID           Dataset repo ID (default: local/sim_pick_vase)"
    echo "  --num-gpus N           Number of GPUs (default: auto-detect)"
    echo "  --batch-size N         Per-GPU batch size (default: 32)"
    echo "  --num-workers N        DataLoader workers (default: 2)"
    echo "  --steps N              Training steps (default: 10000)"
    echo "  --chunk-size N         ACT chunk size (default: 100)"
    echo "  --n-action-steps N     Action steps (default: 50)"
    echo "  --n-obs-steps N        Observation horizon (default: 1)"
    echo "  --save-freq N          Checkpoint frequency (default: 5000)"
    echo "  --seed N               Random seed (default: 42)"
    exit 1
fi

# Auto-detect GPUs if not specified
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    echo "Auto-detected $NUM_GPUS GPUs"
fi

# Verify numpy cache exists
if [ ! -f "/dev/shm/images.npy" ]; then
    echo "ERROR: /dev/shm/images.npy not found"
    echo "Run prepare_dataset.py first:"
    echo "  python prepare_dataset.py --dataset-root $DATASET_ROOT --output-dir /dev/shm"
    exit 1
fi

echo "Training config:"
echo "  GPUs:           $NUM_GPUS"
echo "  Batch size:     $BATCH_SIZE per GPU, $((BATCH_SIZE * NUM_GPUS)) effective"
echo "  Workers:        $NUM_WORKERS"
echo "  Steps:          $STEPS"
echo "  Chunk size:     $CHUNK_SIZE"
echo "  Action steps:   $N_ACTION_STEPS"
echo "  Obs steps:      $N_OBS_STEPS"
echo "  Dataset:        $DATASET_ROOT"
echo ""

accelerate launch \
    --multi_gpu \
    --num_processes=$NUM_GPUS \
    --mixed_precision=bf16 \
    $(which lerobot-train) \
    --policy.type=act \
    --dataset.root=$DATASET_ROOT \
    --dataset.repo_id=$REPO_ID \
    --dataset.image_transforms.enable=false \
    --policy.chunk_size=$CHUNK_SIZE \
    --policy.n_action_steps=$N_ACTION_STEPS \
    --policy.n_obs_steps=$N_OBS_STEPS \
    --policy.push_to_hub=false \
    --batch_size=$BATCH_SIZE \
    --num_workers=$NUM_WORKERS \
    --steps=$STEPS \
    --save_freq=$SAVE_FREQ \
    --eval_freq=0 \
    --log_freq=$LOG_FREQ \
    --seed=$SEED \
    $EXTRA_ARGS
