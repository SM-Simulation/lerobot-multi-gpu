# lerobot-multi-gpu

Fast multi-GPU training for [LeRobot](https://github.com/huggingface/lerobot) ACT policies. Patches lerobot 0.4.4 to bypass the slow HuggingFace parquet dataset path with pre-extracted numpy arrays loaded into shared memory.

**Result:** 0.06 steps/s -> 12+ steps/s on 8x RTX 5090 (200x speedup).

## Branches

| Branch | What |
|--------|------|
| `main` | Speed patches only (n_obs_steps=1) |
| `feature/obs-horizon-support` | Speed patches + observation horizon support (n_obs_steps > 1) |

## The Problem

LeRobot's default data pipeline reads images from parquet files through the HuggingFace `datasets` library. With ACT's action chunking (`chunk_size=100`), every single `__getitem__` call queries the parquet dataset 100 times for action indices. On multi-GPU setups, this makes the DataLoader the bottleneck — GPUs sit idle 95%+ of the time.

## What This Does

### Speed Patches (all branches)

| Patch | File | What |
|-------|------|------|
| `01_fast_numpy_dataset.patch` | `lerobot/datasets/lerobot_dataset.py` | Replaces `__getitem__` to load from pre-extracted numpy arrays in shared memory. Handles action chunking and obs horizon delta_indices entirely from numpy — zero parquet access. Uses `share_memory_()` tensors so forked DataLoader workers get zero-copy access. |
| `02_training_optimizations.patch` | `lerobot/scripts/lerobot_train.py` | Sets `find_unused_parameters=False`, adds `persistent_workers=True`, increases `prefetch_factor`, adds `torch.compile` (auto-disabled when n_obs_steps > 1). |

### Obs Horizon Patches (`feature/obs-horizon-support` branch)

| Patch | File | What |
|-------|------|------|
| `03_act_model_all_fixes.patch` | `lerobot/policies/act/modeling_act.py` | `.item()` -> `.detach()` for torch.compile compatibility. Adds multi-timestep observation support: stacks image features across time, expands pos embeddings to batch dim, handles variable-length state tokens. `@torch.compiler.disable` on inner forward when obs horizon active. |
| `04_obs_horizon_config.patch` | `lerobot/policies/act/configuration_act.py` | Removes the `n_obs_steps != 1` validation error. Adds `observation_delta_indices` property that returns `[-(n_obs-1), ..., 0]` when n_obs_steps > 1. |

## Quick Start

```bash
# 1. Install lerobot
pip install lerobot==0.4.4

# 2. Apply patches
git clone https://github.com/SM-Simulation/lerobot-multi-gpu.git
cd lerobot-multi-gpu

# For obs horizon support:
git checkout feature/obs-horizon-support

bash setup.sh

# 3. Convert your dataset to numpy arrays
cp -r /path/to/dataset /dev/shm/sim_pick_vase
python prepare_dataset.py --dataset-root /dev/shm/sim_pick_vase --output-dir /dev/shm

# 4. Train
bash train.sh --dataset-root /dev/shm/sim_pick_vase
```

## Applying Patches Manually (e.g. on a local workstation)

If `setup.sh` doesn't detect your Python path correctly, apply patches manually:

```bash
# Find your lerobot install
LEROBOT_DIR=$(python3 -c "import lerobot; print(lerobot.__path__[0])")
cd $(dirname $LEROBOT_DIR)

# Apply all patches
for patch in /path/to/lerobot-multi-gpu/patches/*.patch; do
    patch -p0 < "$patch"
done
```

**Important for inference machines:** If you only need to run a trained policy (not train), you only need:
- `04_obs_horizon_config.patch` — allows n_obs_steps > 1
- `03_act_model_all_fixes.patch` — the model forward pass changes

You do NOT need the numpy dataset or training optimization patches for inference.

## Observation Horizon (n_obs_steps > 1)

When training with `--n-obs-steps 2`, the policy receives **2 timesteps** of observations at each step — the current frame and the previous frame. This gives the policy temporal context.

### Training

```bash
bash train.sh --dataset-root /dev/shm/sim_pick_vase \
    --chunk-size 16 --n-action-steps 8 --n-obs-steps 2
```

The dataset pipeline automatically stacks observations using `observation_delta_indices = [-1, 0]`, so no changes to data collection are needed.

### Inference

At inference time, **you must maintain an observation history buffer** and stack observations before passing to the policy:

**Expected input shapes with n_obs_steps=2:**
- `observation.images.realsense`: `(1, 2, C, H, W)` — previous + current frame
- `observation.state`: `(1, 2, 14)` — previous + current state

**Implementation:**
1. Keep a per-environment buffer storing the previous frame's image and state
2. On each policy call, stack `[prev_obs, current_obs]` along dim=1
3. On the **first frame** of an episode (no history), duplicate current obs for both slots
4. Reset the history buffer on episode boundaries

```python
# Example inference pseudocode
prev_obs = {}  # per-env history

def get_action(env_id, current_img, current_state):
    if env_id not in prev_obs:
        # First frame — duplicate current obs
        prev_obs[env_id] = (current_img.clone(), current_state.clone())

    prev_img, prev_state = prev_obs[env_id]

    obs = {
        "observation.images.realsense": torch.stack([prev_img, current_img], dim=1),  # (1, 2, C, H, W)
        "observation.state": torch.stack([prev_state, current_state], dim=1),          # (1, 2, 14)
    }

    action = policy.select_action(preprocessor(obs))

    # Update history
    prev_obs[env_id] = (current_img.clone(), current_state.clone())
    return action

def on_episode_reset(env_id):
    prev_obs.pop(env_id, None)
```

**Note:** `select_action` still returns a single action `(1, 14)`. Action chunking and the action queue work identically to n_obs_steps=1.

## Full Setup on a Fresh VM

```bash
# Install lerobot
pip install lerobot==0.4.4

# Clone and apply patches
git clone https://github.com/SM-Simulation/lerobot-multi-gpu.git
cd lerobot-multi-gpu
git checkout feature/obs-horizon-support  # if you need obs horizon
bash setup.sh

# Copy dataset to RAM disk (fastest I/O)
cp -r ~/.cache/huggingface/lerobot/local/sim_pick_vase /dev/shm/sim_pick_vase

# Extract numpy arrays
python prepare_dataset.py --dataset-root /dev/shm/sim_pick_vase --output-dir /dev/shm

# Train on all available GPUs
bash train.sh --dataset-root /dev/shm/sim_pick_vase

# Train with obs horizon
bash train.sh --dataset-root /dev/shm/sim_pick_vase --n-obs-steps 2 --chunk-size 16 --n-action-steps 8
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
  --n-obs-steps N        Observation horizon (default: 1)
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

**Note:** `torch.compile` is automatically disabled when `n_obs_steps > 1` due to dynamic shape incompatibility with einops. This reduces speed by ~30% but is necessary for correctness.

## Reverting

```bash
bash setup.sh --revert
```

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
| + n_obs_steps=2 (no compile) | ~8 | ~20 min |
