#!/usr/bin/env python3
"""
Boundary Analysis Script

Processes boundary CSV files from neural network training into JSON keyframe format
for animated decision boundary visualization.

Usage:
    python boundary-analysis.py <experiment_name> <data_path> [options]

Example:
    python boundary-analysis.py xor data/baselines/xor --output-dir ../../static/data/nn
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def export_json(data: dict, output_path: str) -> None:
    """Export dictionary to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f)
    print(f"Exported: {output_path}")


def load_iteration_dirs(base_path: Path, max_iters: int = 10) -> list[Path]:
    """Find all iteration directories."""
    dirs = []
    for i in range(1, max_iters + 1):
        iter_path = base_path / f"iter_{i:03d}"
        if iter_path.exists():
            dirs.append(iter_path)
    return dirs


def process_boundary_data(iter_path: Path, iteration_id: str, keyframe_interval: int = 200) -> dict:
    """
    Process boundary CSV files into keyframe format.

    Args:
        iter_path: Path to iteration directory
        iteration_id: e.g. "001"
        keyframe_interval: Sample every N epochs (default 200 = ~10 keyframes for 2000 epochs)

    Returns:
        {
            "iteration": "001",
            "grid_size": 100,
            "keyframes": [
                {"epoch": 0, "data": [...]},
                {"epoch": 200, "data": [...]},
                ...
            ]
        }
    """
    boundary_dir = iter_path / "boundary"
    if not boundary_dir.exists():
        return None

    # Find all epoch files
    epoch_files = sorted(boundary_dir.glob("epoch_*.csv"))
    if not epoch_files:
        return None

    # Get max epoch to include final state
    max_epoch = max(int(f.stem.split("_")[1]) for f in epoch_files)

    keyframes = []
    grid_size = None

    for epoch_file in epoch_files:
        epoch = int(epoch_file.stem.split("_")[1])

        # Include first, last, and every Nth keyframe
        if epoch != 0 and epoch != max_epoch and epoch % keyframe_interval != 0:
            continue

        df = pd.read_csv(epoch_file)

        if grid_size is None:
            grid_size = int(np.sqrt(len(df)))

        # Round predictions to 4 decimal places to reduce file size
        predictions = df['prediction'].round(4).tolist()

        keyframes.append({
            "epoch": epoch,
            "data": predictions
        })

    return {
        "iteration": iteration_id,
        "grid_size": grid_size,
        "keyframes": keyframes
    }


def main():
    parser = argparse.ArgumentParser(description="Process boundary data for web visualization")
    parser.add_argument("experiment", help="Experiment name (e.g., 'xor', 'circle')")
    parser.add_argument("data_path", help="Path to training data directory with iter_XXX subdirs")
    parser.add_argument("--output-dir", default="../../static/data/nn", help="Output directory")
    parser.add_argument("--keyframe-interval", type=int, default=200,
                        help="Sample every N epochs (default: 200)")
    parser.add_argument("--max-iters", type=int, default=10,
                        help="Maximum number of iterations to process")

    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    experiment = args.experiment

    print(f"\nProcessing {experiment} boundary data...")
    print(f"  Source: {data_path}")
    print(f"  Output: {output_dir / experiment / 'boundaries'}")
    print(f"  Keyframe interval: {args.keyframe_interval} epochs")

    iter_dirs = load_iteration_dirs(data_path, args.max_iters)
    print(f"  Found {len(iter_dirs)} iterations")

    for iter_dir in iter_dirs:
        iteration_id = iter_dir.name.split("_")[1]
        boundary_data = process_boundary_data(iter_dir, iteration_id, args.keyframe_interval)

        if boundary_data:
            output_path = output_dir / experiment / "boundaries" / f"{iteration_id}.json"
            export_json(boundary_data, str(output_path))
            print(f"  {iteration_id}: {len(boundary_data['keyframes'])} keyframes")
        else:
            print(f"  {iteration_id}: No boundary data found")

    print("\nDone!")


if __name__ == "__main__":
    main()
