from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

try:
    import torch
    from models.oceanembed_model import OceanEmbedModel
except Exception as exc:
    torch = None
    OceanEmbedModel = None
    _TORCH_IMPORT_ERROR = str(exc)

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "models" / "checkpoints" / "best_model.pt"
PROCESSED = ROOT / "data" / "processed"
DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 5.0, 30.0, 45.0, 105.0


def demo_profile(lat: float, lon: float, day: date):
    """Deterministic fallback so the website is immediately demoable."""
    doy = day.timetuple().tm_yday
    seasonal = math.sin(2 * math.pi * (doy - 30) / 365.25)
    lat_factor = (lat - 17.5) / 12.5
    lon_factor = (lon - 75.0) / 30.0
    surface = 28.0 - 0.10 * lat_factor + 0.35 * lon_factor + 0.65 * seasonal
    temps = []
    for d in DEPTHS:
        cooling = 0.015 * d + 5.0 * (1.0 - math.exp(-d / 120.0))
        temps.append(round(surface - cooling, 4))
    embedding = [
        round(math.sin((i + 1) * 0.17 + lat * 0.03 + lon * 0.01 + doy * 0.005), 6)
        for i in range(64)
    ]
    return temps, embedding


def find_nearest_sample(lat: float, lon: float, day: date):
    best = None
    best_dist = float("inf")
    for split in ("test", "val", "train"):
        directory = PROCESSED / split
        if not directory.exists():
            continue
        for path in sorted(directory.glob("shard_*.npz")):
            try:
                with np.load(path, allow_pickle=False) as data:
                    meta = json.loads(str(data["metadata"].item()))
                    for idx, sample in enumerate(meta.get("samples", [])):
                        if str(sample.get("date", ""))[:10] != day.isoformat():
                            continue
                        dlat = float(sample["latitude"]) - lat
                        dlon = float(sample["longitude"]) - lon
                        dist = dlat*dlat + dlon*dlon
                        if dist < best_dist:
                            best_dist = dist
                            best = (path, idx, sample)
            except Exception:
                continue
    return best


def real_prediction(lat: float, lon: float, day: date):
    if torch is None or OceanEmbedModel is None or not CHECKPOINT.exists():
        return None

    found = find_nearest_sample(lat, lon, day)
    if found is None:
        return None

    path, idx, metadata = found
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    channels = int(checkpoint.get("in_channels", 5))
    embedding_dim = int(checkpoint.get("embedding_dim", 64))
    output_depths = int(checkpoint.get("output_depths", 15))

    with np.load(path, allow_pickle=False) as data:
        x = data["X"][idx].astype(np.float32)

    if x.ndim != 3 or x.shape[-1] != channels:
        raise RuntimeError(
            f"Checkpoint expects {channels} channels but sample has shape {x.shape}."
        )

    x = np.transpose(x, (2, 0, 1))[None, ...]
    model = OceanEmbedModel(
        in_channels=channels,
        embedding_dim=embedding_dim,
        output_dim=output_depths,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        embedding, prediction = model(torch.from_numpy(x))

    depths = checkpoint.get("depths_m", DEPTHS)
    return (
        prediction[0].numpy().round(4).tolist(),
        embedding[0].numpy().round(6).tolist(),
        depths,
        metadata,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    if not (LAT_MIN <= args.lat <= LAT_MAX and LON_MIN <= args.lon <= LON_MAX):
        raise ValueError("Coordinate outside OceanEmbed region.")

    day = date.fromisoformat(args.date)
    real = real_prediction(args.lat, args.lon, day)

    if real is not None:
        temps, embedding, depths, metadata = real
        result = {
            "lat": args.lat, "lon": args.lon, "date": args.date,
            "depths": depths, "temperatures": temps, "embedding": embedding,
            "mode": "trained-model",
            "matched_lat": metadata.get("latitude"),
            "matched_lon": metadata.get("longitude"),
        }
    else:
        temps, embedding = demo_profile(args.lat, args.lon, day)
        result = {
            "lat": args.lat, "lon": args.lon, "date": args.date,
            "depths": DEPTHS, "temperatures": temps, "embedding": embedding,
            "mode": "demo",
            "message": "Demo mode. Add best_model.pt and processed NPZ shards for real inference.",
        }

    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), flush=True)
        sys.exit(1)
