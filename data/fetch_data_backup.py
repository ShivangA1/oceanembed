"""
OceanEmbed Phase 1 — Ocean data acquisition.

Downloads daily ocean fields, then converts them to the model grid:
    latitude:  5..30 N at 0.25°
    longitude: 45..105 E at 0.25°

Required packages:
    pip install copernicusmarine pyyaml xarray netCDF4 numpy

Authentication:
    copernicusmarine login

Expected config:
    configs/config.yaml

The script intentionally downloads GLORYS at its native 1/12° grid and
then regrids locally to 0.25°. The Copernicus subsetter does not provide a
generic "0.25 degree" output-resolution switch, so local regridding is the
safe way to guarantee the requested model grid.

Output:
    data/raw/thetao/YYYY-MM-DD.nc
    data/raw/sss/YYYY-MM-DD.nc
    data/raw/uo/YYYY-MM-DD.nc
    data/raw/vo/YYYY-MM-DD.nc
    data/raw/zos/YYYY-MM-DD.nc
    data/raw/sst/YYYY-MM-DD.nc
"""

from __future__ import annotations

import logging
import shutil
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
import copernicusmarine


CONFIG_PATH = Path("configs/config.yaml")

GLORYS_DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
SST_DATASET = "cmems_obs-sst_glo_phy-temp_nrt_P1D-m"
SSS_DATASET = "cmems_obs-mob_glo_phy-sss_nrt_multi_P1D"

DEFAULT_DEPTHS = [
    0, 5, 10, 20, 30, 50, 75, 100,
    125, 150, 200, 300, 500, 700, 1000,
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("oceanembed")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CONFIG_PATH}. Create it before running this script."
        )
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def cfg_value(cfg, *keys, default=None):
    """Read a nested config value without crashing if an optional key is absent."""
    current = cfg
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def target_grid(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    region = cfg["region"]
    resolution = float(cfg_value(cfg, "region", "resolution", default=0.25))

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


def run_subset(
    dataset_id: str,
    variables: list[str],
    start_date: date,
    end_date: date,
    output_file: Path,
    cfg: dict,
    depth_min: float | None = None,
    depth_max: float | None = None,
) -> bool:
    """Download one small subset with the Copernicus Marine Python API."""
    region = cfg["region"]
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading dataset=%s variables=%s date=%s",
        dataset_id,
        variables,
        start_date,
    )

    try:
        kwargs = dict(
            dataset_id=dataset_id,
            variables=variables,
            minimum_longitude=float(region["lon_min"]),
            maximum_longitude=float(region["lon_max"]),
            minimum_latitude=float(region["lat_min"]),
            maximum_latitude=float(region["lat_max"]),
            start_datetime=f"{start_date.isoformat()}T00:00:00",
            end_datetime=f"{end_date.isoformat()}T23:59:59",
            output_directory=str(output_file.parent),
            output_filename=output_file.name,
            force_download=True,
        )

        if depth_min is not None:
            kwargs["minimum_depth"] = depth_min
            kwargs["maximum_depth"] = depth_max

        copernicusmarine.subset(**kwargs)

        if not output_file.exists():
            logger.error("Download returned but file is missing: %s", output_file)
            return False

        return True

    except Exception:
        logger.exception(
            "Download failed | dataset=%s | variables=%s | date=%s",
            dataset_id,
            variables,
            start_date,
        )
        return False


def find_coord(ds: xr.Dataset, candidates: list[str]) -> str:
    for name in candidates:
        if name in ds.coords:
            return name
        if name in ds.dims:
            return name
    raise KeyError(f"Could not find any coordinate from {candidates}")


def regrid_to_target(
    input_file: Path,
    output_file: Path,
    cfg: dict,
    requested_depths: list[float] | None = None,
    variable: str | None = None,
) -> None:
    """Select requested depths and interpolate the native grid to 0.25°."""
    target_lat, target_lon = target_grid(cfg)

    with xr.open_dataset(input_file) as raw:
        ds = raw.load()

    lat_name = find_coord(ds, ["latitude", "lat", "y"])
    lon_name = find_coord(ds, ["longitude", "lon", "x"])

    # Normalise longitude/latitude coordinate names.
    rename = {}
    if lat_name != "latitude":
        rename[lat_name] = "latitude"
    if lon_name != "longitude":
        rename[lon_name] = "longitude"
    ds = ds.rename(rename)

    # Sort coordinates so interpolation behaves deterministically.
    ds = ds.sortby("latitude").sortby("longitude")

    if requested_depths is not None:
        depth_name = find_coord(ds, ["depth", "deptht", "z"])
        if depth_name != "depth":
            ds = ds.rename({depth_name: "depth"})

        # GLORYS has standard model levels rather than every requested depth.
        # Nearest-neighbour selection maps each requested depth to the closest
        # available GLORYS level and records that mapping in attributes.
        available = ds["depth"].values.astype(float)
        selected = []
        actual = []

        for requested in requested_depths:
            idx = int(np.abs(available - requested).argmin())
            selected.append(float(available[idx]))
            actual.append((float(requested), float(available[idx])))

        ds = ds.sel(depth=selected)
        ds = ds.assign_coords(depth=np.asarray(requested_depths, dtype=float))
        ds.attrs["requested_depths_m"] = ",".join(map(str, requested_depths))
        ds.attrs["source_depth_levels_m"] = ",".join(
            f"{req}->{src}" for req, src in actual
        )

    ds = ds.interp(
        latitude=xr.DataArray(target_lat, dims="latitude"),
        longitude=xr.DataArray(target_lon, dims="longitude"),
        method="linear",
    )

    if variable:
        keep = [variable]
        existing = [v for v in keep if v in ds.data_vars]
        if existing:
            ds = ds[existing]

    ds.attrs["OceanEmbed_grid_resolution_deg"] = float(
        cfg_value(cfg, "region", "resolution", default=0.25)
    )
    ds.attrs["OceanEmbed_note"] = (
        "Regridded locally from the source dataset using xarray linear interpolation."
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    encoding = {}
    for name in ds.data_vars:
        encoding[name] = {"zlib": True, "complevel": 4}

    ds.to_netcdf(output_file, encoding=encoding)


def process_variable(
    name: str,
    dataset_id: str,
    source_variable: str,
    day: date,
    output_dir: Path,
    cfg: dict,
    depth_min: float | None = None,
    depth_max: float | None = None,
    requested_depths: list[float] | None = None,
    regrid: bool = True,
) -> bool:
    final_file = output_dir / name / f"{day}.nc"
    if final_file.exists():
        logger.info("Already exists: %s", final_file)
        return True

    tmp_dir = output_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / f"{name}_{day}.nc"

    try:
        ok = run_subset(
            dataset_id=dataset_id,
            variables=[source_variable],
            start_date=day,
            end_date=day,
            output_file=tmp_file,
            cfg=cfg,
            depth_min=depth_min,
            depth_max=depth_max,
        )
        if not ok:
            return False

        if regrid:
            regrid_to_target(
                tmp_file,
                final_file,
                cfg,
                requested_depths=requested_depths,
                variable=source_variable,
            )
        else:
            final_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp_file), str(final_file))

        logger.info("Saved: %s", final_file)
        return True

    except Exception:
        logger.exception("Processing failed for %s on %s", name, day)
        return False
    finally:
        if tmp_file.exists():
            tmp_file.unlink(missing_ok=True)


def main():
    cfg = load_config()

    start = date.fromisoformat(cfg["date_range"]["start"])
    end = date.fromisoformat(cfg["date_range"]["end"])
    raw_dir = Path(cfg_value(cfg, "output", "raw_dir", default="data/raw"))

    depths = cfg.get("depths", DEFAULT_DEPTHS)
    depths = [float(x) for x in depths]

    sst_variable = cfg_value(
        cfg, "copernicus", "variables", "sst", default="analysed_sst"
    )
    sss_variable = cfg_value(
        cfg, "copernicus", "variables", "sss", default="sos"
    )

    logger.info("=== OceanEmbed Phase 1 ===")
    logger.info("Date range: %s -> %s", start, end)
    logger.info("GLORYS dataset: %s", GLORYS_DATASET)
    logger.info("SST dataset: %s", SST_DATASET)
    logger.info("SSS dataset: %s", SSS_DATASET)
    logger.info("Target depths: %s", depths)
    logger.info("Target grid: 0.25°")

    skipped = {
        "thetao": [],
        "sss": [],
        "uo": [],
        "vo": [],
        "zos": [],
        "sst": [],
    }

    for day in date_range(start, end):
        logger.info("========== %s ==========", day)

        # 1. 3-D temperature target.
        if not process_variable(
            "thetao",
            GLORYS_DATASET,
            "thetao",
            day,
            raw_dir,
            cfg,
            depth_min=min(depths),
            depth_max=max(depths),
            requested_depths=depths,
        ):
            skipped["thetao"].append(str(day))

        # 2. Surface salinity from the dedicated Copernicus multi-observation
        # product. It is already a surface field, so no depth is requested.
        if not process_variable(
            "sss",
            SSS_DATASET,
            sss_variable,
            day,
            raw_dir,
            cfg,
            requested_depths=None,
        ):
            skipped["sss"].append(str(day))

        # 3. Surface currents from GLORYS.
        # GLORYS surface model level is ~0.494 m, so request 0..1 m.
        if not process_variable(
            "uo",
            GLORYS_DATASET,
            "uo",
            day,
            raw_dir,
            cfg,
            depth_min=0.0,
            depth_max=1.0,
        ):
            skipped["uo"].append(str(day))

        if not process_variable(
            "vo",
            GLORYS_DATASET,
            "vo",
            day,
            raw_dir,
            cfg,
            depth_min=0.0,
            depth_max=1.0,
        ):
            skipped["vo"].append(str(day))

        # 4. Sea surface height from GLORYS.
        if not process_variable(
            "zos",
            GLORYS_DATASET,
            "zos",
            day,
            raw_dir,
            cfg,
            requested_depths=None,
        ):
            skipped["zos"].append(str(day))

        # 5. Independent SST input.
        if not process_variable(
            "sst",
            SST_DATASET,
            sst_variable,
            day,
            raw_dir,
            cfg,
            requested_depths=None,
        ):
            skipped["sst"].append(str(day))

    print("\n==============================")
    print("DOWNLOAD SUMMARY")
    print("==============================")

    total_skipped = 0
    for variable, dates in skipped.items():
        print(f"\n{variable}:")
        if dates:
            for skipped_date in dates:
                print(f"  SKIPPED: {skipped_date}")
            total_skipped += len(dates)
        else:
            print("  No skipped days.")

    print(f"\nTotal skipped downloads: {total_skipped}")
    print(f"Output directory: {raw_dir.resolve()}")


if __name__ == "__main__":
    main()