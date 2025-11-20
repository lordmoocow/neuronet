#!/usr/bin/env python3
"""
Generate circle classification dataset for neural network training.

Creates random 2D points and labels them as inside (1) or outside (0) a circle.
Perfect for testing neural network generalization capabilities.
"""

import random
import sys

def generate_circle_data(n_samples, center_x=0.5, center_y=0.5, radius=0.3, seed=None):
    """
    Generate random points and classify them as inside/outside a circle.

    Args:
        n_samples: Number of samples to generate
        center_x: Circle center x-coordinate (default: 0.5)
        center_y: Circle center y-coordinate (default: 0.5)
        radius: Circle radius (default: 0.3)
        seed: Random seed for reproducibility (optional)

    Returns:
        List of (x, y, label) tuples where label is 1 if inside circle, 0 otherwise
    """
    if seed is not None:
        random.seed(seed)

    data = []
    for _ in range(n_samples):
        # Generate random point in [0, 1] x [0, 1]
        x = random.random()
        y = random.random()

        # Calculate distance from circle center
        distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5

        # Label: 1 if inside circle, 0 if outside
        label = 1.0 if distance < radius else 0.0

        data.append((x, y, label))

    return data

def save_to_csv(data, filename):
    """Save data to CSV file (no header)."""
    with open(filename, 'w') as f:
        for x, y, label in data:
            f.write(f"{x},{y},{label}\n")

def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_circle_data.py <output_file> <n_samples> [seed]")
        print()
        print("Examples:")
        print("  python generate_circle_data.py train.csv 100      # Generate 100 training samples")
        print("  python generate_circle_data.py test.csv 1000 42   # Generate 1000 test samples with seed 42")
        sys.exit(1)

    output_file = sys.argv[1]
    n_samples = int(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else None

    # Generate data (circle centered at 0.5, 0.5 with radius 0.3)
    data = generate_circle_data(n_samples, seed=seed)

    # Save to CSV
    save_to_csv(data, output_file)

    # Print statistics
    inside = sum(1 for _, _, label in data if label == 1.0)
    outside = n_samples - inside

    print(f"Generated {n_samples} samples:")
    print(f"  Inside circle:  {inside} ({inside/n_samples*100:.1f}%)")
    print(f"  Outside circle: {outside} ({outside/n_samples*100:.1f}%)")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
