#!/usr/bin/env python3
"""Extract a LeRobot v3 dataset into fast numpy arrays for multi-GPU training.

Reads a standard LeRobot dataset (parquet + images) and creates flat numpy arrays
that can be memory-mapped or loaded into shared memory for zero-overhead data loading.

Usage:
    # After converting episodes with convert_to_lerobot.py:
    python prepare_dataset.py --dataset-root /path/to/dataset --output-dir /dev/shm

    # Or copy dataset to /dev/shm first for fastest I/O:
    cp -r ~/.cache/huggingface/lerobot/local/sim_pick_vase /dev/shm/sim_pick_vase
    python prepare_dataset.py --dataset-root /dev/shm/sim_pick_vase --output-dir /dev/shm
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


def extract_dataset(dataset_root: Path, output_dir: Path) -> None:
    """Extract a LeRobot v3 dataset into flat numpy arrays."""
    info_path = dataset_root / "meta" / "info.json"
    assert info_path.exists(), f"Not a LeRobot dataset: {info_path} not found"

    with open(info_path) as f:
        info = json.load(f)

    total_frames = info["total_frames"]
    features = info["features"]
    print(f"Dataset: {total_frames} frames, {info['total_episodes']} episodes")

    # Identify image keys and their shapes
    image_keys = []
    for key, feat in features.items():
        if feat["dtype"] == "image":
            image_keys.append((key, tuple(feat["shape"])))
    print(f"Image keys: {[k for k, _ in image_keys]}")

    # Read all parquet files
    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    assert len(parquet_files) > 0, f"No parquet files found in {data_dir}"
    print(f"Found {len(parquet_files)} parquet files")

    # Load all parquet data
    tables = []
    for pf in parquet_files:
        tables.append(pq.read_table(pf))
    import pyarrow as pa
    table = pa.concat_tables(tables)
    n = len(table)
    assert n == total_frames, f"Expected {total_frames} frames, got {n}"

    # Extract scalar/vector columns
    print("Extracting scalar columns...")
    col_map = {
        "observation.state": ("states", "float32"),
        "action": ("actions", "float32"),
        "index": ("indices", "int64"),
        "episode_index": ("episode_indices", "int64"),
        "frame_index": ("frame_indices", "int64"),
        "timestamp": ("timestamps", "float32"),
        "task_index": ("task_indices", "int64"),
    }

    for col_name, (out_name, dtype) in col_map.items():
        if col_name in table.column_names:
            arr = table.column(col_name).to_pandas().values
            if arr.dtype == object:
                # Nested lists — stack them
                arr = np.stack(arr).astype(dtype)
            else:
                arr = arr.astype(dtype)
            out_path = output_dir / f"{out_name}.npy"
            np.save(out_path, arr)
            print(f"  {out_name}: {arr.shape} {arr.dtype} -> {out_path}")

    # Extract images
    for image_key, shape in image_keys:
        safe_name = image_key.replace(".", "_").replace("/", "_")
        print(f"Extracting images for '{image_key}' (shape {shape})...")
        h, w, c = shape

        # Check if images are stored inline (bytes) or as file paths
        col = table.column(image_key)
        sample = col[0].as_py()

        if isinstance(sample, dict) and "bytes" in sample and sample["bytes"]:
            # Image stored as inline bytes (most common for lerobot image datasets)
            import io
            images = np.zeros((n, h, w, c), dtype=np.uint8)
            for i in tqdm(range(n), desc=f"  Loading {image_key}"):
                entry = col[i].as_py()
                img = np.array(Image.open(io.BytesIO(entry["bytes"])))
                images[i] = img[:h, :w, :c]
        elif isinstance(sample, dict) and "path" in sample:
            # Image stored as file reference
            images = np.zeros((n, h, w, c), dtype=np.uint8)
            for i in tqdm(range(n), desc=f"  Loading {image_key}"):
                entry = col[i].as_py()
                img_path = dataset_root / entry["path"]
                img = np.array(Image.open(img_path))
                images[i] = img[:h, :w, :c]
        else:
            raise ValueError(f"Unknown image format for {image_key}: {type(sample)}")

        out_path = output_dir / "images.npy"
        np.save(out_path, images)
        print(f"  images: {images.shape} {images.dtype} -> {out_path}")
        print(f"  Size: {images.nbytes / 1e9:.1f} GB")

    print(f"\nDone! All arrays saved to {output_dir}/")
    print("Files:")
    for f in sorted(output_dir.glob("*.npy")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract LeRobot dataset to fast numpy arrays"
    )
    parser.add_argument(
        "--dataset-root", type=str, required=True,
        help="Path to LeRobot dataset root (containing meta/info.json)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="/dev/shm",
        help="Output directory for numpy arrays (default: /dev/shm)"
    )
    args = parser.parse_args()
    extract_dataset(Path(args.dataset_root), Path(args.output_dir))
