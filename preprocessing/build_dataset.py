"""
OceanEmbed Phase 2 — preprocessing and patch dataset builder.

Inputs expected under:
    data/raw/sst/YYYY-MM-DD.nc
    data/raw/sss/YYYY-MM-DD.nc
    data/raw/zos/YYYY-MM-DD.nc
    data/raw/uo/YYYY-MM-DD.nc
    data/raw/vo/YYYY-MM-DD.nc
    data/raw/winds/YYYY-MM-DD.nc   # optional for now
    data/raw/thetao/YYYY-MM-DD.nc

The builder:
- harmonizes all fields to the configured 0.25° grid
- aligns dates
- preserves land as NaN
- fills only small coastal NaN gaps
- computes train-only z-score statistics
- extracts 9x9 patches lazily, one time step at a time
- splits by time, never randomly
- writes compressed NPZ shards

IMPORTANT:
Winds are optional. If data/raw/winds is absent, the script builds a
6-channel dataset (sst, sss, zos, uo, vo, plus the target thetao).
Once winds are added, it automatically builds the requested 7-channel
input tensor (sst, sss, zos, uo, vo, u10, v10).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

CONFIG_PATH = Path("configs/config.yaml")
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SCALER_PATH = Path("preprocessing/scaler_stats.json")

PATCH_SIZE = 9
HALF_PATCH = PATCH_SIZE // 2

BASE_CHANNELS = ["sst", "sss", "zos", "uo", "vo"]
WIND_CHANNELS = ["u10", "v10"]
TARGET_NAME = "thetao"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("oceanembed.preprocessing")


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_region(cfg: dict):
    region = cfg["region"]
    resolution = float(region.get("resolution", 0.25))

    lat = np.arange(
        float(region["lat_min"]),
        float(region["lat_max"]) + resolution / 2,
        resolution,
    )
    lon = np.arange(
        float(region["lon_min"]),
        float(region["lon_max"]) + resolution / 2,
        resolution,
    )
    return lat, lon


def find_coord(ds: xr.Dataset, candidates):
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"Could not find coordinate among {candidates}")


def normalise_coords(ds: xr.Dataset) -> xr.Dataset:
    lat_name = find_coord(ds, ["latitude", "lat", "y"])
    lon_name = find_coord(ds, ["longitude", "lon", "x"])

    rename = {}
    if lat_name != "latitude":
        rename[lat_name] = "latitude"
    if lon_name != "longitude":
        rename[lon_name] = "longitude"

    ds = ds.rename(rename)
    return ds.sortby("latitude").sortby("longitude")


def choose_data_var(ds: xr.Dataset, preferred: str | None = None) -> str:
    if preferred and preferred in ds.data_vars:
        return preferred

    # Common aliases returned by Copernicus/CDS products.
    aliases = {
        "sst": ["analysed_sst", "sst", "sea_surface_temperature"],
        "sss": ["sos", "sss", "so"],
        "zos": ["zos", "sla", "adt"],
        "uo": ["uo", "u"],
        "vo": ["vo", "v"],
        "u10": ["u10", "10m_u_component_of_wind"],
        "v10": ["v10", "10m_v_component_of_wind"],
        "thetao": ["thetao", "temperature"],
    }

    for alias in aliases.get(preferred or "", []):
        if alias in ds.data_vars:
            return alias

    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))

    raise KeyError(
        f"Could not identify variable {preferred!r}. "
        f"Available variables: {list(ds.data_vars)}"
    )


def reduce_to_surface(ds: xr.Dataset) -> xr.Dataset:
    """Select the shallowest available depth and remove depth metadata."""
    if "depth" in ds.coords or "depth" in ds.dims:
        depth_values = ds["depth"].values.astype(float)
        shallowest = float(depth_values[np.argmin(np.abs(depth_values))])
        ds = ds.sel(depth=shallowest, method="nearest")

        # Critical: remove the scalar depth coordinate.
        if "depth" in ds.coords:
            ds = ds.drop_vars("depth")

    return ds


def reduce_temperature(ds: xr.Dataset, expected_depths) -> xr.Dataset:
    """Ensure target is exactly depth x latitude x longitude."""
    if "depth" not in ds.coords and "depth" not in ds.dims:
        raise ValueError("thetao file has no depth coordinate")

    ds = ds.sortby("depth")

    # fetch_data_fixed.py labels requested depths after mapping them to
    # available GLORYS levels. If the labels are present, use them directly.
    available = ds["depth"].values.astype(float)
    selected = []
    for d in expected_depths:
        idx = int(np.abs(available - float(d)).argmin())
        selected.append(float(available[idx]))

    ds = ds.sel(depth=selected)
    ds = ds.assign_coords(depth=np.asarray(expected_depths, dtype=np.float32))
    return ds


def load_one_file(path: Path, name: str, expected_depths=None) -> xr.DataArray:
    with xr.open_dataset(path) as raw:
        ds = raw.load()

    ds = normalise_coords(ds)

    var = choose_data_var(ds, name)

    if name == TARGET_NAME:
        ds = reduce_temperature(ds, expected_depths)
    else:
        ds = reduce_to_surface(ds)

    da = ds[var]

    # Collapse any accidental singleton dimensions except time/depth/lat/lon.
    for dim in list(da.dims):
        if dim not in {"time", "depth", "latitude", "longitude"}:
            if da.sizes[dim] == 1:
                da = da.isel({dim: 0}, drop=True)

    if "time" not in da.dims:
        # Daily files should represent one day. Give them a synthetic time
        # from the filename if no time coordinate survived the source file.
        try:
            stamp = datetime.fromisoformat(path.stem)
            da = da.expand_dims(time=[np.datetime64(stamp.date())])
        except ValueError:
            da = da.expand_dims(time=[np.datetime64("NaT")])

    return da


def target_grid(cfg):
    lat, lon = get_region(cfg)
    return xr.Dataset(
        coords={
            "latitude": ("latitude", lat),
            "longitude": ("longitude", lon),
        }
    )


def regrid_da(da: xr.DataArray, grid: xr.Dataset) -> xr.DataArray:
    """
    Bilinear regrid to the common target grid.

    xarray-regrid is used when installed. xarray.interp is the compatible
    fallback and performs the same linear/bilinear interpolation for a
    rectilinear latitude/longitude grid.
    """
    try:
        import xarray_regrid  # noqa: F401

        # The package API has changed across releases. Use its standard
        # regrid method when available.
        if hasattr(da, "regrid"):
            return da.regrid(grid, method="bilinear")
    except Exception:
        pass

    return da.interp(
        latitude=grid.latitude,
        longitude=grid.longitude,
        method="linear",
    )


def date_from_file(path: Path) -> np.datetime64:
    return np.datetime64(path.stem)


def available_dates(name: str):
    directory = RAW_DIR / name
    if not directory.exists():
        return {}

    result = {}
    for path in sorted(directory.glob("*.nc")):
        try:
            result[date_from_file(path)] = path
        except Exception:
            logger.warning("Ignoring non-date file: %s", path)
    return result


def common_dates(channel_names):
    maps = [available_dates(name) for name in channel_names]
    if not maps:
        return []

    common = set(maps[0])
    for m in maps[1:]:
        common &= set(m)

    return sorted(common)


def optional_winds_present() -> bool:
    return bool(available_dates("winds"))


def load_day(
    day,
    channel_names,
    target_depths,
    grid,
):
    arrays = []

    for name in channel_names:
        if name in WIND_CHANNELS:
            path = available_dates("winds")[day]
        else:
            path = available_dates(name)[day]

        da = load_one_file(path, name, target_depths)
        da = regrid_da(da, grid)
        da = da.squeeze(drop=True)

        if "time" in da.dims:
            da = da.isel(time=0, drop=True)

        arrays.append(da)

    target_path = available_dates(TARGET_NAME)[day]
    target = load_one_file(target_path, TARGET_NAME, target_depths)
    target = regrid_da(target, grid).squeeze(drop=True)

    if "time" in target.dims:
        target = target.isel(time=0, drop=True)

    # Inputs: lat x lon x channel.
    input_stack = xr.concat(
        arrays,
        dim=xr.IndexVariable("channel", channel_names),
        coords="minimal",
        compat="override",
    ).transpose("latitude", "longitude", "channel")

    # Target: depth x lat x lon.
    target = target.transpose("depth", "latitude", "longitude")

    return input_stack, target


def fill_small_coastal_gaps(arr: np.ndarray, max_radius: int = 1) -> np.ndarray:
    """
    Fill only NaN cells adjacent to valid ocean cells.

    This deliberately performs at most one-cell nearest-neighbour filling.
    It does NOT iteratively fill large land masses.
    """
    out = arr.copy()

    if out.ndim != 3:
        raise ValueError("Expected lat x lon x channels array")

    h, w, c = out.shape

    for k in range(c):
        field = out[:, :, k]
        nan = np.isnan(field)

        for _ in range(max_radius):
            if not nan.any():
                break

            padded = np.pad(field, 1, mode="constant", constant_values=np.nan)
            candidates = [
                padded[:-2, 1:-1],
                padded[2:, 1:-1],
                padded[1:-1, :-2],
                padded[1:-1, 2:],
            ]

            filled = field.copy()
            for candidate in candidates:
                mask = np.isnan(filled) & np.isfinite(candidate)
                filled[mask] = candidate[mask]

            field = filled
            nan = np.isnan(field)

        out[:, :, k] = field

    return out


def valid_ocean_mask(inputs: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Ocean cell = all required inputs and all 15 target depths are finite."""
    input_valid = np.all(np.isfinite(inputs), axis=2)
    target_valid = np.all(np.isfinite(target), axis=0)
    return input_valid & target_valid


def compute_train_stats(
    train_days,
    channel_names,
    target_depths,
    grid,
):
    sums = np.zeros(len(channel_names), dtype=np.float64)
    sums_sq = np.zeros(len(channel_names), dtype=np.float64)
    counts = np.zeros(len(channel_names), dtype=np.int64)

    for i, day in enumerate(train_days, 1):
        logger.info("Scaler pass %d/%d: %s", i, len(train_days), day)
        inputs, _ = load_day(day, channel_names, target_depths, grid)
        x = inputs.values.astype(np.float64)

        finite = np.isfinite(x)
        sums += np.where(finite, x, 0).sum(axis=(0, 1))
        sums_sq += np.where(finite, x * x, 0).sum(axis=(0, 1))
        counts += finite.sum(axis=(0, 1))

    if np.any(counts == 0):
        bad = [channel_names[i] for i, n in enumerate(counts) if n == 0]
        raise ValueError(f"No finite training values for channels: {bad}")

    mean = sums / counts
    variance = np.maximum(sums_sq / counts - mean**2, 1e-12)
    std = np.sqrt(variance)

    return mean, std


def split_dates(dates):
    """
    Time-only split:
      first 70% -> train
      next 15%  -> validation
      final 15% -> test
    """
    n = len(dates)
    if n < 3:
        raise ValueError("Need at least 3 dates for train/val/test split.")

    train_end = max(1, int(n * 0.70))
    val_end = max(train_end + 1, int(n * 0.85))
    val_end = min(val_end, n - 1)

    return dates[:train_end], dates[train_end:val_end], dates[val_end:]


def shard_writer(split_name, shard_index, X, y, metadata):
    out_dir = PROCESSED_DIR / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"shard_{shard_index:05d}.npz"
    np.savez_compressed(
        path,
        X=X.astype(np.float32),
        y=y.astype(np.float32),
        metadata=np.array(json.dumps(metadata)),
    )
    logger.info("Wrote %s | X=%s | y=%s", path, X.shape, y.shape)


def extract_split(
    split_name,
    dates,
    channel_names,
    target_depths,
    grid,
    mean,
    std,
    shard_size=2048,
):
    X_buffer = []
    y_buffer = []
    metadata_buffer = []
    shard_index = 0

    for date_index, day in enumerate(dates):
        logger.info(
            "Extracting %s date %d/%d: %s",
            split_name,
            date_index + 1,
            len(dates),
            day,
        )

        inputs_da, target_da = load_day(day, channel_names, target_depths, grid)
        inputs = inputs_da.values.astype(np.float32)
        target = target_da.values.astype(np.float32)

        # Fill only small gaps for input channels. Target NaNs are never filled.
        inputs = fill_small_coastal_gaps(inputs, max_radius=1)

        # Normalize inputs using TRAINING statistics only.
        inputs = (inputs - mean.astype(np.float32)) / std.astype(np.float32)

        # Validity is checked after filling inputs and before extracting.
        mask = valid_ocean_mask(inputs, target)

        valid_positions = np.argwhere(mask)
        h, w, _ = inputs.shape

        for lat_idx, lon_idx in valid_positions:
            if (
                lat_idx < HALF_PATCH
                or lat_idx >= h - HALF_PATCH
                or lon_idx < HALF_PATCH
                or lon_idx >= w - HALF_PATCH
            ):
                continue

            patch = inputs[
                lat_idx - HALF_PATCH : lat_idx + HALF_PATCH + 1,
                lon_idx - HALF_PATCH : lon_idx + HALF_PATCH + 1,
                :,
            ]

            label = target[:, lat_idx, lon_idx]

            if not np.all(np.isfinite(patch)) or not np.all(np.isfinite(label)):
                continue

            X_buffer.append(patch)
            y_buffer.append(label)
            metadata_buffer.append(
                {
                    "date": str(day),
                    "lat_index": int(lat_idx),
                    "lon_index": int(lon_idx),
                    "latitude": float(grid.latitude.values[lat_idx]),
                    "longitude": float(grid.longitude.values[lon_idx]),
                }
            )

            if len(X_buffer) >= shard_size:
                shard_writer(
                    split_name,
                    shard_index,
                    np.stack(X_buffer),
                    np.stack(y_buffer),
                    {
                        "channels": channel_names,
                        "depths_m": target_depths,
                        "patch_size": PATCH_SIZE,
                        "samples": metadata_buffer,
                    },
                )
                shard_index += 1
                X_buffer.clear()
                y_buffer.clear()
                metadata_buffer.clear()

    if X_buffer:
        shard_writer(
            split_name,
            shard_index,
            np.stack(X_buffer),
            np.stack(y_buffer),
            {
                "channels": channel_names,
                "depths_m": target_depths,
                "patch_size": PATCH_SIZE,
                "samples": metadata_buffer,
            },
        )


def main():
    cfg = load_config()
    target_depths = [float(x) for x in cfg.get("depths", [])]

    if len(target_depths) != 15:
        raise ValueError(
            f"Expected exactly 15 target depths, got {len(target_depths)}."
        )

    grid = target_grid(cfg)

    channels = list(BASE_CHANNELS)
    if optional_winds_present():
        channels.extend(WIND_CHANNELS)
        logger.info("Winds detected: using 7 input channels.")
    else:
        logger.warning(
            "No data/raw/winds directory found. "
            "Building 5-channel ocean input dataset for now. "
            "Add winds and rerun to produce the final 7-channel dataset."
        )

    required_for_dates = channels + [TARGET_NAME]
    dates = common_dates(
        [c if c not in WIND_CHANNELS else "winds" for c in required_for_dates]
    )

    if not dates:
        raise RuntimeError(
            "No common dates found across all required raw variables. "
            "Run Phase 1 test downloads first."
        )

    train_dates, val_dates, test_dates = split_dates(dates)

    logger.info("Common dates: %d", len(dates))
    logger.info(
        "Split: train=%d, val=%d, test=%d",
        len(train_dates),
        len(val_dates),
        len(test_dates),
    )

    mean, std = compute_train_stats(train_dates, channels, target_depths, grid)

    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    scaler = {
        "channels": channels,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "training_start": str(train_dates[0]),
        "training_end": str(train_dates[-1]),
        "patch_size": PATCH_SIZE,
        "grid_resolution": float(cfg["region"].get("resolution", 0.25)),
    }

    SCALER_PATH.write_text(
        json.dumps(scaler, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved scaler: %s", SCALER_PATH)

    # Record split boundaries for reproducibility.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    split_metadata = {
        "train": [str(train_dates[0]), str(train_dates[-1])],
        "validation": [str(val_dates[0]), str(val_dates[-1])],
        "test": [str(test_dates[0]), str(test_dates[-1])],
        "channels": channels,
        "depths_m": target_depths,
        "patch_size": PATCH_SIZE,
    }
    (PROCESSED_DIR / "split_metadata.json").write_text(
        json.dumps(split_metadata, indent=2),
        encoding="utf-8",
    )

    shard_size = int(cfg.get("preprocessing", {}).get("shard_size", 2048))

    extract_split(
        "train",
        train_dates,
        channels,
        target_depths,
        grid,
        mean,
        std,
        shard_size,
    )
    extract_split(
        "val",
        val_dates,
        channels,
        target_depths,
        grid,
        mean,
        std,
        shard_size,
    )
    extract_split(
        "test",
        test_dates,
        channels,
        target_depths,
        grid,
        mean,
        std,
        shard_size,
    )

    logger.info("Phase 2 preprocessing complete.")


if __name__ == "__main__":
    main()
