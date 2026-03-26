#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
How I ran it on RACC2
1) cd /storage/research/metstudent/msc/users_2026/mg826635/oifs-expt
2) module load anaconda/2023.09-0/base
3) python3 task2_task3_analysis.py

"""

from pathlib import Path
import glob
import csv
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# Evaluation domain (Minerva box)
LAT_N, LAT_S = 55, 35
LON_W, LON_E = 0, 40

# Forecast NetCDF paths (T+48)
CONTROL_NC = "/storage/research/metstudent/msc/users_2026/mg826635/oifs-expt/2023051500_control/netcdf/control_surface_T48.nc"
NORAD_NC   = "/storage/research/metstudent/msc/users_2026/mg826635/oifs-expt/2023051500_norad/netcdf/norad_surface_T48.nc"

# Analysis NetCDF path (valid at same time)
ANALYSIS_NC = "/storage/research/metstudent/OpenIFS/analysis/minerva/2023051700/ICMGGaby6+000000_surface_1R.nc"

# Minerva time-lagged ensemble (valid at same time)
ENSEMBLE_ROOT = "/storage/research/metstudent/OpenIFS/ensembles/minerva"
ENSEMBLE_STARTS = [ "2023051300", "2023051306", "2023051312", "2023051318","2023051400", "2023051406", "2023051412", "2023051418",]

# Output CSV filename (written in current working directory)
OUT_CSV = "task2_task3_summary.csv"


# HELPERS

def _ensure_exists(path: str | Path, label: str) -> Path:
    """Return Path if it exists, else raise FileNotFoundError with a clear message."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def _select_box(da: xr.DataArray) -> xr.DataArray:
    """
    Select the evaluation box from a DataArray with latitude/longitude coordinates.

    Notes:
    - ECMWF/OpenIFS latitude is typically descending (90 -> -90), so slice(55,35) is correct.
    - Longitudes are assumed 0..360; the box uses 0..40E.
    """
    return da.sel(latitude=slice(LAT_N, LAT_S), longitude=slice(LON_W, LON_E))



# TASK 2: PRECIPITATION

def mean_rainfall_mm(nc_path: str | Path) -> float:
    
    """Compute area-mean total precipitation (tp) in mm over the evaluation box."""
    
    p = _ensure_exists(nc_path, "Forecast NetCDF")
    ds = xr.open_dataset(p)
    
    # quick sanity check (time should be 2023-05-17 for T+48)
    # print(ds["time"].values)

    if "tp" not in ds:
        raise KeyError(f"'tp' not found in {p.name}. Available: {list(ds.data_vars)}")

    tp_mm = ds["tp"].isel(time=0) * 1000.0  # m -> mm
    return float(_select_box(tp_mm).mean().values)


def ensemble_rainfall_mm() -> dict[str, float]:
    
    """Compute area-mean tp (mm) for each Minerva time-lagged ensemble member"""
    
    root = Path(ENSEMBLE_ROOT)
    if not root.exists():
        raise FileNotFoundError(f"Ensemble root not found: {root}")

    results: dict[str, float] = {}
    for start in ENSEMBLE_STARTS:
        pattern = str(root / start / "*surface_1R.nc")
        matches = glob.glob(pattern)

        # On HPC we expect exactly one file per start time.
        if len(matches) != 1:
            print(f"[WARN] {start}: expected 1 match for {pattern}, got {len(matches)} -> skipping")
            continue

        results[start] = mean_rainfall_mm(matches[0])

    return results



# TASK 3: MSLP RMSE

def msl_rmse_hpa(forecast_nc: str | Path, analysis_nc: str | Path = ANALYSIS_NC, region: str = "global") -> float:
    
    """ RMSE of mean sea-level pressure (msl) in hPa against analysis."""
    
    fc_p = _ensure_exists(forecast_nc, "Forecast NetCDF")
    an_p = _ensure_exists(analysis_nc, "Analysis NetCDF")

    fc = xr.open_dataset(fc_p)["msl"].isel(time=0) / 100.0  # Pa -> hPa
    an = xr.open_dataset(an_p)["msl"].isel(time=0) / 100.0  # Pa -> hPa

    if region == "box":
        fc = _select_box(fc)
        an = _select_box(an)
    elif region != "global":
        raise ValueError("region must be 'global' or 'box'")

    return float(np.sqrt(((fc - an) ** 2).mean()).values)



# OUTPUT
def write_summary_csv(rows: list[tuple[str, float | None, str, str]], out_csv: str | Path) -> None:
    """Write summary rows to CSV."""
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "value", "units", "notes"])
        for item, value, units, notes in rows:
            w.writerow([item, "" if value is None else f"{value:.6f}", units, notes])



# MAIN

def main() -> None:
    # Task 2 
    control_tp = mean_rainfall_mm(CONTROL_NC)
    norad_tp   = mean_rainfall_mm(NORAD_NC)

    ens_dict = None
    ens_min = ens_mean = ens_max = None

    # Ensemble is optional context; do not hard-fail if missing.
    try:
        tmp = ensemble_rainfall_mm()
        if len(tmp) > 0:
            ens_dict = tmp
            vals = np.array(list(ens_dict.values()), dtype=float)
            ens_min, ens_mean, ens_max = float(vals.min()), float(vals.mean()), float(vals.max())
        else:
            print("[INFO] Ensemble present but no members matched expected pattern -> ensemble stats skipped.")
    except FileNotFoundError:
        print("[INFO] Ensemble root not found -> skipping ensemble context.")
    except Exception as e:
        print(f"[INFO] Ensemble processing skipped due to: {e}")
        

    # Task 3
    control_rmse_global = control_rmse_box = None
    norad_rmse_global = norad_rmse_box = None

    try:
        control_rmse_global = msl_rmse_hpa(CONTROL_NC, ANALYSIS_NC, region="global")
        control_rmse_box    = msl_rmse_hpa(CONTROL_NC, ANALYSIS_NC, region="box")
        norad_rmse_global   = msl_rmse_hpa(NORAD_NC, ANALYSIS_NC, region="global")
        norad_rmse_box      = msl_rmse_hpa(NORAD_NC, ANALYSIS_NC, region="box")
    except FileNotFoundError:
        print("[INFO] Analysis file not found -> skipping RMSE verification.")
    except Exception as e:
        print(f"[INFO] RMSE verification skipped due to: {e}")

    # Print summary 
    print("\n=== TASK 2: Rainfall (tp, mm) at T+48 valid 2023-05-17 00Z ===")
    print(f"Region box: {LAT_S}–{LAT_N}N, {LON_W}–{LON_E}E")
    print(f"Control: {control_tp:.3f} mm")
    print(f"NoRad:   {norad_tp:.3f} mm")
    if ens_dict is not None:
        print(f"Ensemble: min {ens_min:.3f} | mean {ens_mean:.3f} | max {ens_max:.3f} (n={len(ens_dict)})")
    else:
        print("Ensemble: (skipped / unavailable)")

    print("\n=== TASK 3: MSLP RMSE (hPa) vs Analysis at 2023-05-17 00Z ===")
    if control_rmse_global is not None:
        print(f"Control RMSE: global {control_rmse_global:.3f} | box {control_rmse_box:.3f}")
        print(f"NoRad   RMSE: global {norad_rmse_global:.3f} | box {norad_rmse_box:.3f}")
    else:
        print("RMSE verification: (skipped / unavailable)")

    # Write CSV summary 
    rows: list[tuple[str, float | None, str, str]] = [
        ("control_tp", control_tp, "mm", "Area-mean tp in Minerva box"),
        ("norad_tp", norad_tp, "mm", "Area-mean tp in Minerva box"),
        ("ens_min_tp", ens_min, "mm", "Min of ensemble members (if available)"),
        ("ens_mean_tp", ens_mean, "mm", "Mean of ensemble members (if available)"),
        ("ens_max_tp", ens_max, "mm", "Max of ensemble members (if available)"),
    ]

    if ens_dict is not None:
        for start, val in sorted(ens_dict.items()):
            rows.append((f"ens_{start}_tp", val, "mm", "Ensemble member area-mean tp"))

    rows.extend([
        ("control_rmse_global", control_rmse_global, "hPa", "RMSE of msl over full grid"),
        ("control_rmse_box", control_rmse_box, "hPa", "RMSE of msl over Minerva box"),
        ("norad_rmse_global", norad_rmse_global, "hPa", "RMSE of msl over full grid"),
        ("norad_rmse_box", norad_rmse_box, "hPa", "RMSE of msl over Minerva box"),
    ])

    write_summary_csv(rows, OUT_CSV)
    print(f"\nWrote: {OUT_CSV}\n")
    
    
    
# TASK 4: Linear regression between rainfall and SSR

def run_task4():
    from netCDF4 import Dataset
    import numpy as np

    def simple_linregress(x, y):
        m, c = np.polyfit(x, y, 1)
        y_fit = m * x + c
        ss_res = np.sum((y - y_fit)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot
        return m, c, r2

    norad_file = "/storage/research/metstudent/msc/users_2026/mg826635/oifs-expt/2023051500_norad/netcdf/norad_surface_T48.nc"
    control_file = "/storage/research/metstudent/msc/users_2026/mg826635/oifs-expt/2023051500_control/netcdf/control_surface_T48.nc"

    def run_case(label, path):
        ds = Dataset(path)

        rain = ds.variables["crr"][0,:,:] + ds.variables["lsrr"][0,:,:]
        ssr  = ds.variables["ssr"][0,:,:]
        lsm  = ds.variables["lsm"][0,:,:]

        x = ssr.flatten()
        y = rain.flatten()
        l = lsm.flatten()

        mask = np.isfinite(x) & np.isfinite(y)
        x, y, l = x[mask], y[mask], l[mask]

        print(f"\n=== {label} ===")

        m, c, r2 = simple_linregress(x, y)
        plt.figure(figsize=(5,4))
        plt.scatter(x, y, s=1, alpha=0.3)
        plt.plot(x, m*x + c)

        plt.xlabel("Surface Shortwave Radiation (ssr)")
        plt.ylabel("Rain Rate (crr + lsrr)")
        plt.title(f"{label}: Rain vs SSR (R²={r2:.4f})")

        plt.savefig(f"{label}_regression.png", dpi=300)
        plt.close()
        
        print("All points:")
        print("  slope =", m)
        print("  r^2   =", r2)

        sea = l < 0.5
        land = l >= 0.5

        if sea.any():
            m, c, r2 = simple_linregress(x[sea], y[sea])
            print("Sea points:")
            print("  slope =", m)
            print("  r^2   =", r2)

        if land.any():
            m, c, r2 = simple_linregress(x[land], y[land])
            print("Land points:")
            print("  slope =", m)
            print("  r^2   =", r2)

    run_case("CONTROL", control_file)
    run_case("NORAD", norad_file)

if __name__ == "__main__":
    main()          # Tasks 2 & 3
    run_task4()     # Task 4