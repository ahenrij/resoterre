"""
Utilities for preprocessing raw RDPS data into the format required for inference.

Preprocessing pipeline (matching training):
  1. Load variables from pre-regridded RDPS_regrid files  (256×512)
     OR regrid on-the-fly from raw RDPS using scipy zoom (fallback)
  2. Anomaly variables have already had the climatology subtracted in
     RDPS_regrid/{var}_anomaly/ — or we subtract it here from the
     RDPS_climatology/ files when starting from raw data.
  3. Normalize per variable specifications (rdps_variables.py).
  4. Load static fields (topography MF, land-sea mask sftlf) at 2km (1024×2048).
  5. Assemble and save the preprocessed NetCDF batch.

File naming conventions (observed from reference data):
  Raw RDPS     : {path_rdps}/{YYYYMMDDCC}_{FFF}.nc
                   CC = run cycle (00), FFF = forecast hour (007…012)
  Regridded    : {path_rdps_regrid}/{var_name}/{YYYYMMDDHH}.nc
  Anomaly reg. : {path_rdps_regrid}/{var_name}_anomaly/{YYYYMMDDHH}.nc
  Climatology  : {path_rdps_climatology}/{var_name}/rdps_climatology_{var}_{MM-DD}T{HH}.nc
  Static MF    : {path_hrdps_mf}          (single file, var: MF)
  Static sftlf : {path_hrdps_sftlf}       (single file, var: HRDPS_sftlf)
"""

import re
from pathlib import Path
import numpy as np
from scipy.ndimage import zoom
import xarray
from resoterre.data_management.netcdf_utils import CFVariables
from resoterre.datasets.rdps.rdps_variables import rdps_variables as RDPS_VARIABLE_SPECS
from resoterre.ml.data_loader_utils import normalize


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def parse_raw_rdps_filename(filename: str) -> tuple[np.datetime64, int]:
    """
    Parse a raw RDPS filename into run time and forecast hour.

    Parameters
    ----------
    filename : str
        E.g. '2024010100_007.nc'

    Returns
    -------
    tuple[np.datetime64, int]
        (run_time, forecast_hour)
    """
    stem = Path(filename).stem  # e.g. '2024010100_007'
    m = re.fullmatch(r"(\d{8})(\d{2})_(\d{3})", stem)
    if not m:
        raise ValueError(f"Unrecognised raw RDPS filename: {filename}")
    date_str, cycle_str, fhour_str = m.group(1), m.group(2), m.group(3)
    run_time = np.datetime64(
        f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T{cycle_str}:00:00"
    )
    return run_time, int(fhour_str)


def valid_time_from_raw_filename(filename: str) -> np.datetime64:
    """Return the valid time (run_time + forecast_hour) from a raw RDPS filename."""
    run_time, fhour = parse_raw_rdps_filename(filename)
    return run_time + np.timedelta64(fhour, "h")


def valid_time_to_regrid_stem(valid_time: np.datetime64) -> str:
    """
    Convert a valid time to the regridded filename stem.

    E.g. np.datetime64('2024-01-01T07') → '2024010107'
    """
    dt = valid_time.astype("datetime64[s]").astype(object)
    return f"{dt.year:04d}{dt.month:02d}{dt.day:02d}{dt.hour:02d}"


def climatology_filename(var_name: str, valid_time: np.datetime64) -> str:
    """
    Build the climatology filename for a given variable and valid time.

    Pattern: rdps_climatology_{var}_{MM-DD}T{HH}.nc
    E.g. 'rdps_climatology_GZ500_01-01T08.nc'
    """
    dt = valid_time.astype("datetime64[s]").astype(object)
    return f"rdps_climatology_{var_name}_{dt.month:02d}-{dt.day:02d}T{dt.hour:02d}.nc"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _read_first_spatial_var(ds: xarray.Dataset) -> np.ndarray:
    """
    Return the data of the first data variable as a 2D (lat, lon) array.

    Squeezes out time, level, and pressure singleton dimensions.
    """
    for var_name in ds.data_vars:
        data = ds[var_name].values.squeeze()
        if data.ndim == 2:
            return data
        if data.ndim == 3:
            # Pick index 0 along the leading dim (time or level)
            return data[0]
    raise ValueError(f"No 2D or 3D spatial variable found in dataset. vars={list(ds.data_vars)}")


def load_from_regrid_files(
    path_rdps_regrid: Path,
    var_name: str,
    valid_time: np.datetime64,
    is_anomaly: bool,
) -> np.ndarray:
    """
    Load a pre-regridded variable from RDPS_regrid files.

    Returns a 2D array of shape (256, 512).

    Parameters
    ----------
    path_rdps_regrid : Path
        Root directory, e.g. '.../RDPS_regrid'.
    var_name : str
        Config variable name, e.g. 'GZ500', 'TT_model_levels', 'UU850'.
    valid_time : np.datetime64
        Valid datetime to look up.
    is_anomaly : bool
        If True, look in the '{var_name}_anomaly' sub-directory.
    """
    folder_name = f"{var_name}_anomaly" if is_anomaly else var_name
    stem = valid_time_to_regrid_stem(valid_time)
    file_path = path_rdps_regrid / folder_name / f"{stem}.nc"

    if not file_path.exists():
        raise FileNotFoundError(f"Regridded file not found: {file_path}")

    ds = xarray.open_dataset(file_path, decode_timedelta=False)
    data = _read_first_spatial_var(ds)
    ds.close()
    return data.astype(np.float32)


def load_from_raw_rdps(
    raw_file: Path,
    var_name: str,
) -> np.ndarray:
    """
    Extract a variable from a raw RDPS file on the native (1076×1102) grid.

    Parameters
    ----------
    raw_file : Path
        Raw RDPS NetCDF file.
    var_name : str
        Config variable name (e.g., 'GZ500', 'HU850', 'TT_model_levels').
    """
    ds = xarray.open_dataset(raw_file, decode_timedelta=False)

    if var_name[-3:].isdigit():
        # Pressure-level variable e.g. GZ500, HU850, UU850, VV850
        base = var_name[:-3]          # 'GZ', 'HU', 'UU', 'VV'
        pressure = float(var_name[-3:])
        # Try {base}_pressure_levels first, then bare base
        nc_var = f"{base}_pressure_levels" if f"{base}_pressure_levels" in ds else base
        if nc_var not in ds:
            raise ValueError(f"Variable {nc_var} not found in {raw_file}")
        pres_levels = ds["pres"].values
        pres_idx = int(np.argmin(np.abs(pres_levels - pressure)))
        data = ds[nc_var].isel(pres=pres_idx).values.squeeze()
    else:
        # Model-level or surface variable
        if var_name not in ds:
            raise ValueError(f"Variable {var_name} not found in {raw_file}")
        data = ds[var_name].values.squeeze()

    ds.close()
    return data.astype(np.float32)


def load_climatology_field(
    path_rdps_climatology: Path,
    var_name: str,
    valid_time: np.datetime64,
) -> np.ndarray | None:
    """
    Load the climatology field for a variable at a given valid time.

    Returns a 2D array on the native RDPS grid (1076×1102), or None if not found.

    Filename pattern: rdps_climatology_{var}_{MM-DD}T{HH}.nc
    Variable inside file: first data variable.
    """
    clim_file = path_rdps_climatology / var_name / climatology_filename(var_name, valid_time)
    if not clim_file.exists():
        return None

    ds = xarray.open_dataset(clim_file, decode_timedelta=False)
    data = _read_first_spatial_var(ds)
    ds.close()
    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# Regridding (scipy zoom — fast approximation)
# ---------------------------------------------------------------------------

def regrid_field(data: np.ndarray, target_h: int, target_w: int, order: int = 1) -> np.ndarray:
    """
    Resize a 2D field to (target_h, target_w) using bilinear interpolation.

    Used as fallback when pre-regridded files are not available.
    The production pipeline uses xESMF nearest_s2d weights (see path_grids).

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (h, w).
    target_h, target_w : int
        Target dimensions.
    order : int
        Interpolation order (1 = bilinear).
    """
    if data.shape == (target_h, target_w):
        return data
    return zoom(data, [target_h / data.shape[0], target_w / data.shape[1]], order=order).astype(data.dtype)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_variable(data: np.ndarray, var_name: str, is_anomaly: bool = False) -> np.ndarray:
    """
    Normalize a variable to [-1, 1] using pre-defined specifications.

    Uses the '_anomaly' spec when is_anomaly=True (narrower range).

    Parameters
    ----------
    data : np.ndarray
        Input field.
    var_name : str
        Variable name as in rdps_variables.py.
    is_anomaly : bool
        Whether this is an anomaly field.
    """
    lookup = f"{var_name}_anomaly" if is_anomaly and f"{var_name}_anomaly" in RDPS_VARIABLE_SPECS else var_name

    if lookup not in RDPS_VARIABLE_SPECS:
        print(f"    Warning: No normalization spec for '{lookup}', using data min/max")
        return normalize(data, mode=(-1, 1))

    spec = RDPS_VARIABLE_SPECS[lookup]
    if spec.normalize_min is None or spec.normalize_max is None:
        print(f"    Warning: normalize_min/max not set for '{lookup}', using data min/max")
        return normalize(data, mode=(-1, 1))

    return normalize(
        data,
        mode=(-1, 1),
        valid_min=spec.normalize_min,
        valid_max=spec.normalize_max,
        log_normalize=spec.log_normalize,
        log_offset=spec.normalize_log_offset if spec.normalize_log_offset is not None else 1.0,
    )


# ---------------------------------------------------------------------------
# Static fields (input_last_layer)
# ---------------------------------------------------------------------------

def load_static_fields(
    path_hrdps_mf: Path | None,
    path_hrdps_sftlf: Path | None,
    h_out: int,
    w_out: int,
) -> np.ndarray:
    """
    Load and normalize the two static fields that form input_last_layer.

    Channel 0 — Topography (MF): var 'MF' in HRDPS_MF_2km_north_america_ml.nc
    Channel 1 — Land-sea mask (sftlf): var 'HRDPS_sftlf' in HRDPS_sftlf_2km_north_america_ml.nc

    Both files are already on the 2km grid (1024×2048).
    If a file is missing, the corresponding channel is filled with zeros and
    a warning is printed.

    Returns
    -------
    np.ndarray
        Shape (2, h_out, w_out), float32.
    """
    static = np.zeros((2, h_out, w_out), dtype=np.float32)

    # Channel 0 – topography (MF)
    if path_hrdps_mf is not None and Path(path_hrdps_mf).exists():
        ds = xarray.open_dataset(path_hrdps_mf, decode_timedelta=False)
        mf = ds["MF"].values.squeeze().astype(np.float32)  # (1024, 2048)
        ds.close()
        if mf.shape != (h_out, w_out):
            mf = regrid_field(mf, h_out, w_out)
        # Normalize: clip & scale elevation to [-1, 1]
        # Spec: range roughly [-5, 5000] m, normalize over [0, 4000]
        static[0] = normalize(mf, mode=(-1, 1), valid_min=0.0, valid_max=4000.0)
    else:
        print("  Warning: path_hrdps_mf not found – topography channel set to zeros")

    # Channel 1 – land-sea mask (sftlf), already in [0, 1]
    if path_hrdps_sftlf is not None and Path(path_hrdps_sftlf).exists():
        ds = xarray.open_dataset(path_hrdps_sftlf, decode_timedelta=False)
        sftlf = ds["HRDPS_sftlf"].values.squeeze().astype(np.float32)  # (1024, 2048)
        ds.close()
        if sftlf.shape != (h_out, w_out):
            sftlf = regrid_field(sftlf, h_out, w_out)
        # Normalize [0, 1] → [-1, 1]
        static[1] = normalize(sftlf, mode=(-1, 1), valid_min=0.0, valid_max=1.0)
    else:
        print("  Warning: path_hrdps_sftlf not found – land-sea mask channel set to zeros")

    return static


# ---------------------------------------------------------------------------
# Main batch creation
# ---------------------------------------------------------------------------

def create_preprocessed_batch(
    raw_rdps_file: Path,
    rdps_variables: list[str],
    hrdps_variables: list[str],
    anomaly_variables: list[str],
    output_path: Path,
    h_in: int = 256,
    w_in: int = 512,
    h_out: int = 1024,
    w_out: int = 2048,
    path_rdps_regrid: Path | None = None,
    path_rdps_climatology: Path | None = None,
    path_hrdps_mf: Path | None = None,
    path_hrdps_sftlf: Path | None = None,
) -> str:
    """
    Create a single preprocessed batch from one raw RDPS file.

    For each variable:
      - If path_rdps_regrid exists and the file is found → load from regrid files (fast).
      - Otherwise → extract from raw RDPS → subtract climatology if needed
        → regrid with scipy zoom (approximate).
    Then normalizes and assembles the output NetCDF.

    Parameters
    ----------
    raw_rdps_file : Path
        Raw RDPS file, e.g. '.../2024010100_007.nc'.
    rdps_variables : list[str]
        10 input channel names matching model training order.
    hrdps_variables : list[str]
        4 output variable names.
    anomaly_variables : list[str]
        Subset of rdps_variables to be treated as anomalies.
    output_path : Path
        Destination NetCDF file.
    h_in, w_in : int
        Input spatial dimensions (8km grid: 256×512).
    h_out, w_out : int
        Output spatial dimensions (2km grid: 1024×2048).
    path_rdps_regrid : Path or None
        Root of pre-regridded variable directories.
    path_rdps_climatology : Path or None
        Root of climatology directories.
    path_hrdps_mf : Path or None
        Path to the HRDPS topography NetCDF file.
    path_hrdps_sftlf : Path or None
        Path to the HRDPS land-sea mask NetCDF file.

    Returns
    -------
    str
        Path to the saved preprocessed batch file.
    """
    # Determine valid time from the raw filename
    valid_time = valid_time_from_raw_filename(raw_rdps_file.name)
    n_input_channels = len(rdps_variables)
    n_target_channels = len(hrdps_variables)

    print(f"  File     : {raw_rdps_file.name}")
    print(f"  Valid at : {valid_time}")
    print(f"  Input grid  (8km): ({h_in}, {w_in})")
    print(f"  Output grid (2km): ({h_out}, {w_out})")

    input_data = np.zeros((1, n_input_channels, h_in, w_in), dtype=np.float32)

    for i, var_name in enumerate(rdps_variables):
        is_anomaly = var_name in anomaly_variables

        # Load
        loaded_from_regrid = False
        if path_rdps_regrid is not None:
            try:
                field = load_from_regrid_files(path_rdps_regrid, var_name, valid_time, is_anomaly)
                loaded_from_regrid = True
                source = "regrid files"
            except FileNotFoundError:
                pass  # fall through to raw loading

        if not loaded_from_regrid:
            field = load_from_raw_rdps(raw_rdps_file, var_name)  # native grid

            # Subtract climatology before regridding (on native grid)
            if is_anomaly:
                clim = load_climatology_field(path_rdps_climatology, var_name, valid_time) \
                    if path_rdps_climatology else None
                if clim is not None:
                    field = field - clim
                else:
                    print(f"    Warning: no climatology for '{var_name}' at {valid_time}")

            field = regrid_field(field, h_in, w_in)
            source = "raw+zoom"

        # --- Normalize ---
        normalized = normalize_variable(field, var_name, is_anomaly=is_anomaly)
        input_data[0, i] = normalized

        tag = "anomaly" if is_anomaly else "raw"
        print(f"  [{i+1:2d}/{n_input_channels}] {var_name} ({tag}, {source}): "
              f"range=[{field.min():.3g}, {field.max():.3g}] → "
              f"norm=[{normalized.min():.3g}, {normalized.max():.3g}]")

    # --- Static fields ---
    print(f"\n  Loading static fields...")
    static = load_static_fields(path_hrdps_mf, path_hrdps_sftlf, h_out, w_out)
    input_last_layer = static[np.newaxis, :, :, :]  # (1, 2, h_out, w_out)

    # Target: zeros (not needed for inference)
    target = np.zeros((1, n_target_channels, h_out, w_out), dtype=np.float32)

    # --- Spatial / temporal coordinates ---
    dt = valid_time.astype("datetime64[s]").astype(object)

    # 1D lat/lon from a loaded regridded file, or fallback to linspace
    lat_1d = np.linspace(18.26, 79.64, h_out, dtype=np.float32)  # 2km grid extent
    lon_1d = np.linspace(-158.14, -35.32, w_out, dtype=np.float32)

    # Attempt to read exact coordinates from a regrid file
    if path_rdps_regrid is not None:
        try:
            stem = valid_time_to_regrid_stem(valid_time)
            any_var = rdps_variables[0]
            is_anom = any_var in anomaly_variables
            folder = f"{any_var}_anomaly" if is_anom else any_var
            ref_file = path_rdps_regrid / folder / f"{stem}.nc"
            if ref_file.exists():
                ds_ref = xarray.open_dataset(ref_file, decode_timedelta=False)
                # These are 8km coords; we'll re-use linspace for 2km output
                ds_ref.close()
        except Exception:
            pass

    # --- Build CF dataset ---
    cf_coords = CFVariables()
    cf_coords.add("height_in_idx",  dims=("sample", "height_in"),  data=np.arange(h_in,  dtype=np.int16)[np.newaxis], dtype=np.int16)
    cf_coords.add("width_in_idx",   dims=("sample", "width_in"),   data=np.arange(w_in,  dtype=np.int16)[np.newaxis], dtype=np.int16)
    cf_coords.add("height_out_idx", dims=("sample", "height_out"), data=np.arange(h_out, dtype=np.int16)[np.newaxis], dtype=np.int16)
    cf_coords.add("width_out_idx",  dims=("sample", "width_out"),  data=np.arange(w_out, dtype=np.int16)[np.newaxis], dtype=np.int16)
    cf_coords.add("lat", dims=("sample", "height_out"), data=lat_1d[np.newaxis], dtype=np.float32,
                  attributes={"units": "degrees_north", "standard_name": "latitude"})
    cf_coords.add("lon", dims=("sample", "width_out"),  data=lon_1d[np.newaxis], dtype=np.float32,
                  attributes={"units": "degrees_east", "standard_name": "longitude"})
    cf_coords.add("year",  dims=("sample",), data=np.array([dt.year],  dtype=np.int16), dtype=np.int16)
    cf_coords.add("month", dims=("sample",), data=np.array([dt.month], dtype=np.int8),  dtype=np.int8)
    cf_coords.add("day",   dims=("sample",), data=np.array([dt.day],   dtype=np.int8),  dtype=np.int8)
    cf_coords.add("hour",  dims=("sample",), data=np.array([dt.hour],  dtype=np.int8),  dtype=np.int8)
    cf_coords.add("input_variables",  dims=("input_channel",),  data=np.array(rdps_variables,  dtype="object"), dtype="object")
    cf_coords.add("output_variables", dims=("target_channel",), data=np.array(hrdps_variables, dtype="object"), dtype="object")

    cf_vars = CFVariables()
    cf_vars.add("input_first_block", dims=("sample", "input_channel",      "height_in",  "width_in"),  data=input_data,       dtype=np.float32, zlib=True, complevel=4)
    cf_vars.add("input_last_layer",  dims=("sample", "last_layer_channel", "height_out", "width_out"), data=input_last_layer, dtype=np.float32, zlib=True, complevel=4)
    cf_vars.add("target",            dims=("sample", "target_channel",     "height_out", "width_out"), data=target,           dtype=np.float32, zlib=True, complevel=4)

    ds_out = xarray.Dataset(
        data_vars=cf_vars,
        coords=cf_coords,
        attrs={"Conventions": "CF-1.6", "description": "Preprocessed RDPS data for downscaling inference"},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(output_path, engine="h5netcdf")

    return str(output_path)
