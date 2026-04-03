# lerobot-multi-gpu

Fast multi-GPU training for [LeRobot](https://github.com/huggingface/lerobot) ACT policies. Patches lerobot 0.4.4 to bypass the slow HuggingFace parquet dataset path with pre-extracted numpy arrays loaded into shared memory.

**Result:** 0.06 steps/s -> 12+ steps/s on 8x RTX 5090 (200x speedup).

## The Problem

LeRobot's default data pipeline reads images from parquet files through the HuggingFace `datasets` library. With ACT's action chunking (`chunk_size=100`), every single `__getitem__` call queries the parquet dataset 100 times for action indices. On multi-GPU setups, this makes the DataLoader the bottleneck — GPUs sit idle 95%+ of the time.

## What This Does

Three patches to lerobot 0.4.4:

| Patch | File | What |
|-------|------|------|
| `01_fast_numpy_dataset.patch` | `lerobot/datasets/lerobot_dataset.py` | Replaces `__getitem__` to load from pre-extracted numpy arrays in shared memory. Handles action chunking (delta_indices) entirely from numpy — zero parquet access. Uses `share_memory_()` tensors so forked DataLoader workers get zero-copy access. |
| `02_training_optimizations.patch` | `lerobot/scripts/lerobot_train.py` | Sets `find_unused_parameters=False` (removes extra autograd traversal), adds `persistent_workers=True`, increases `prefetch_factor`, adds `torch.compile`. |
| `03_act_compile_fix.patch` | `lerobot/policies/act/modeling_act.py` | Replaces `.item()` with `.detach()` in loss dict to prevent `torch.compile` graph breaks. |

## Quick Start

```bash
# 1. Install lerobot
pip install lerobot==0.4.4

# 2. Apply patches
git clone https://github.com/HOCC493/lerobot-multi-gpu.git
cd lerobot-multi-gpu
bash setup.sh

# 3. Convert your dataset to numpy arrays
#    (assumes you already have a LeRobot dataset at /path/to/dataset)
cp -r /path/to/dataset /dev/shm/sim_pick_vase
python prepare_dataset.py --dataset-root /dev/shm/sim_pick_vase --output-dir /dev/shm

# 4. Train
bash train.sh --dataset-root /dev/shm/sim_pick_vase
```

## Full Setup on a Fresh VM

```bash
# Install lerobot
pip install lerobot==0.4.4

# Clone and apply patches
git clone https://github.com/HOCC493/lerobot-multi-gpu.git
cd lerobot-multi-gpu
bash setup.sh

# Copy dataset to RAM disk (fastest I/O)
cp -r ~/.cache/huggingface/lerobot/local/sim_pick_vase /dev/shm/sim_pick_vase

# Extract numpy arrays
python prepare_dataset.py --dataset-root /dev/shm/sim_pick_vase --output-dir /dev/shm

# Train on all available GPUs
bash train.sh --dataset-root /dev/shm/sim_pick_vase

# Train with specific GPU count / batch size
bash train.sh --dataset-root /dev/shm/sim_pick_vase --num-gpus 4 --batch-size 64
```

## Training Options

```
bash train.sh --dataset-root PATH [options]

Options:
  --dataset-root PATH    Path to LeRobot dataset (required)
  --repo-id ID           Dataset repo ID (default: local/sim_pick_vase)
  --num-gpus N           Number of GPUs (default: auto-detect)
  --batch-size N         Per-GPU batch size (default: 32)
  --num-workers N        DataLoader workers (default: 2)
  --steps N              Training steps (default: 10000)
  --chunk-size N         ACT chunk size (default: 100)
  --n-action-steps N     Action steps (default: 50)
  --save-freq N          Checkpoint frequency (default: 5000)
  --seed N               Random seed (default: 42)
```

## Scaling Guidelines

| GPUs | Recommended batch_size | Effective batch | Expected steps/s |
|------|----------------------|-----------------|-----------------|
| 1    | 128                  | 128             | 15-25           |
| 2    | 64                   | 128             | 12-20           |
| 4    | 32-64                | 128-256         | 12-18           |
| 8    | 32                   | 256             | 10-15           |

**Key insight:** For small datasets (<100K frames), fewer GPUs with larger per-GPU batch sizes can be faster than many GPUs with small batches, because NCCL allreduce overhead becomes significant relative to compute.

## Dataset Requirements

The numpy extraction creates these files in `/dev/shm/`:

| File | Shape | Size (17K frames) |
|------|-------|-------------------|
| `images.npy` | `(N, H, W, 3)` uint8 | 3.9 GB |
| `states.npy` | `(N, 14)` float32 | 1 MB |
| `actions.npy` | `(N, 14)` float32 | 1 MB |
| `indices.npy` | `(N,)` int64 | 140 KB |
| `episode_indices.npy` | `(N,)` int64 | 140 KB |
| `frame_indices.npy` | `(N,)` int64 | 140 KB |
| `timestamps.npy` | `(N,)` float32 | 70 KB |
| `task_indices.npy` | `(N,)` int64 | 140 KB |

**RAM requirement:** The images array is loaded into each GPU process's memory as float32 tensors (~15 GB for 17K frames at 240x320). With 8 GPUs, that's ~15 GB shared across all processes via `share_memory_()`. Ensure your system has enough RAM for the dataset + GPU process overhead.

## Reverting

```bash
bash setup.sh --revert
```

This restores the original lerobot files from backups created during setup.

## How It Works

### Before (slow path)
```
DataLoader worker -> __getitem__(idx)
  -> self.hf_dataset[idx]          # parquet decode for main frame
  -> self._query_hf_dataset(...)   # parquet decode x100 for action chunk
  -> PIL decode PNG from parquet bytes
  -> ~16ms per item = 16s per batch of 1024
```

### After (fast path)
```
Main process (once at startup):
  -> np.load('images.npy')         # 3.9 GB into RAM
  -> torch.from_numpy(...).share_memory_()  # shared across all workers

DataLoader worker -> __getitem__(idx)
  -> self._images_tensor[idx]      # tensor slice, no copy
  -> self._actions_tensor[indices]  # tensor slice for action chunk
  -> ~0.15ms per item = 20ms per batch of 128
```

## Benchmarks

System: 8x RTX 5090, 2x AMD EPYC 9B14, 1.1 TiB RAM

| Configuration | Steps/s | Time for 10K steps |
|--------------|---------|-------------------|
| Vanilla lerobot (parquet) | 0.06 | ~46 hours |
| + numpy mmap (no delta fix) | 0.06 | ~46 hours |
| + numpy cache + delta fix | 6.9 | ~24 min |
| + batch_size=32, torch.compile | 12.5 | ~13 min |
