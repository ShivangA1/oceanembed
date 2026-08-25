from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import copernicusmarine
import xarray as xr
import numpy as np
import yaml


CONFIG = Path("configs/config.yaml")

GLORYS_DEFAULT = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
SSS_DEFAULT = "cmems_obs-mob_glo_phy-sss_my_multi_P1D"
SST_DEFAULT = "C3S-GLO-SST-L4-REP-OBS-SST"

DEFAULT_DEPTHS = [
    0, 5, 10, 20, 30, 50, 75, 100,
    125, 150, 200, 300, 500, 700, 1000,
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("oceanembed.phase1")


def load_config():
    with CONFIG.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def days_between(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def subset_one(
    *,
    dataset_id: str,
    variable: str,
    day: date,
    out_file: Path,
    cfg: dict,
    depth_min=None,
    depth_max=None,
):
    region = cfg["region"]
    out_file.parent.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        dataset_id=dataset_id,
        variables=[variable],
        minimum_longitude=float(region["lon_min"]),
        maximum_longitude=float(region["lon_max"]),
        minimum_latitude=float(region["lat_min"]),
        maximum_latitude=float(region["lat_max"]),
        start_datetime=f"{day.isoformat()}T00:00:00",
        end_datetime=f"{day.isoformat()}T23:59:59",
        output_directory=str(out_file.parent),
        output_filename=out_file.name,
        force_download=True,
    )

    if depth_min is not None:
        kwargs["minimum_depth"] = depth_min
    if depth_max is not None:
        kwargs["maximum_depth"] = depth_max

    log.info(
        "dataset=%s | variable=%s | date=%s",
        dataset_id, variable, day
    )
    copernicusmarine.subset(**kwargs)

    if not out_file.exists():
        raise FileNotFoundError(f"Copernicus returned without creating {out_file}")


def process_glorys(
    dataset,
    variable,
    name,
    day,
    cfg,
    depths=None,
    surface=False,
):
    out = Path(cfg["output"]["raw_dir"]) / name / f"{day}.nc"

    if out.exists():
        log.info("Already exists: %s", out)
        return True

    try:
        if surface:
            subset_one(
                dataset_id=dataset,
                variable=variable,
                day=day,
                out_file=out,
                cfg=cfg,
                depth_min=0,
                depth_max=1,
            )
        else:
            subset_one(
                dataset_id=dataset,
                variable=variable,
                day=day,
                out_file=out,
                cfg=cfg,
                depth_min=min(depths),
                depth_max=max(depths),
            )
        return True
    except Exception:
        log.exception("FAILED %s for %s", name, day)
        out.unlink(missing_ok=True)
        return False


def process_surface(
    dataset,
    variable,
    name,
    day,
    cfg,
):
    out = Path(cfg["output"]["raw_dir"]) / name / f"{day}.nc"

    if out.exists():
        log.info("Already exists: %s", out)
        return True

    try:
        subset_one(
            dataset_id=dataset,
            variable=variable,
            day=day,
            out_file=out,
            cfg=cfg,
        )
        return True
    except Exception:
        log.exception("FAILED %s for %s", name, day)
        out.unlink(missing_ok=True)
        return False


def main():
    cfg = load_config()

    region = cfg["region"]
    start = date.fromisoformat(cfg["date_range"]["start"])
    end = date.fromisoformat(cfg["date_range"]["end"])

    depths = [float(x) for x in cfg.get("depths", DEFAULT_DEPTHS)]
    if len(depths) != 15:
        raise ValueError("Expected exactly 15 target depths.")

    glorys = (
        cfg.get("copernicus", {})
        .get("datasets", {})
        .get("physics", GLORYS_DEFAULT)
    )

    sss_cfg = cfg.get("sss", {})
    sss_dataset = sss_cfg.get("dataset", SSS_DEFAULT)
    sss_variable = sss_cfg.get("variable", "sos")

    sst_cfg = cfg.get("sst", {})
    sst_dataset = sst_cfg.get("dataset", SST_DEFAULT)
    sst_variable = sst_cfg.get("variable", "analysed_sst")

    log.info("=== OceanEmbed Phase 1 — 5 channels ===")
    log.info(
        "Region: %.2f..%.2f N, %.2f..%.2f E",
        region["lat_min"], region["lat_max"],
        region["lon_min"], region["lon_max"],
    )
    log.info("Dates: %s -> %s", start, end)
    log.info("GLORYS: %s", glorys)
    log.info("SSS: %s | %s", sss_dataset, sss_variable)
    log.info("SST: %s | %s", sst_dataset, sst_variable)
    log.info("Depths: %s", depths)

    skipped = {
        "thetao": [],
        "uo": [],
        "vo": [],
        "zos": [],
        "sss": [],
        "sst": [],
    }

    for day in days_between(start, end):
        log.info("========== %s ==========", day)

        jobs = [
            (
                "thetao",
                process_glorys(
                    glorys, "thetao", "thetao", day, cfg,
                    depths=depths, surface=False
                ),
            ),
            (
                "uo",
                process_glorys(
                    glorys, "uo", "uo", day, cfg,
                    depths=depths, surface=True
                ),
            ),
            (
                "vo",
                process_glorys(
                    glorys, "vo", "vo", day, cfg,
                    depths=depths, surface=True
                ),
            ),
            (
                "zos",
                process_surface(
                    glorys, "zos", "zos", day, cfg
                ),
            ),
            (
                "sss",
                process_surface(
                    sss_dataset, sss_variable, "sss", day, cfg
                ),
            ),
            (
                "sst",
                process_surface(
                    sst_dataset, sst_variable, "sst", day, cfg
                ),
            ),
        ]

        for name, ok in jobs:
            if not ok:
                skipped[name].append(str(day))

    print("\n==============================")
    print("PHASE 1 SUMMARY")
    print("==============================")

    total = 0
    for name, dates in skipped.items():
        if dates:
            print(f"{name}: SKIPPED {len(dates)} day(s)")
            for d in dates:
                print(f"  {d}")
            total += len(dates)
        else:
            print(f"{name}: OK")

    print(f"\nTotal failed variable-days: {total}")
    print(
        "Raw output:",
        Path(cfg["output"]["raw_dir"]).resolve()
    )


if __name__ == "__main__":
    main()