"""
OceanEmbed Phase 2 — visual sanity check.

Usage:
    python preprocessing/inspect_dataset.py

Requires:
    pip install matplotlib numpy
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROCESSED_DIR = Path("data/processed")
SPLIT = "train"


def load_first_shard():
    shards = sorted((PROCESSED_DIR / SPLIT).glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(
            f"No shards found under {PROCESSED_DIR / SPLIT}. "
            "Run build_dataset.py first."
        )

    with np.load(shards[0], allow_pickle=False) as data:
        X = data["X"]
        y = data["y"]
        metadata = json.loads(str(data["metadata"]))

    return shards[0], X, y, metadata


def main():
    shard, X, y, metadata = load_first_shard()

    channels = metadata["channels"]
    depths = np.asarray(metadata["depths_m"], dtype=float)

    print(f"Shard: {shard}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Channels: {channels}")
    print(f"Depths: {depths.tolist()}")

    # Plot the centre-cell map for the first input channel.
    # A 9x9 patch is shown; the centre is the labelled location.
    sample = X[0]
    channel_index = 0

    plt.figure(figsize=(7, 6))
    plt.imshow(sample[:, :, channel_index], origin="lower")
    plt.axvline(4, linewidth=1)
    plt.axhline(4, linewidth=1)
    plt.title(f"Input channel: {channels[channel_index]}")
    plt.xlabel("Patch longitude index")
    plt.ylabel("Patch latitude index")
    plt.colorbar(label="Normalized value")
    plt.tight_layout()
    plt.show()

    # Plot 3 random temperature profiles.
    rng = np.random.default_rng(42)
    count = min(3, len(y))
    indices = rng.choice(len(y), size=count, replace=False)

    plt.figure(figsize=(7, 6))
    for idx in indices:
        plt.plot(y[idx], depths, marker="o", label=f"sample {idx}")

    plt.gca().invert_yaxis()
    plt.xlabel("Temperature")
    plt.ylabel("Depth (m)")
    plt.title("Random target temperature profiles")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()