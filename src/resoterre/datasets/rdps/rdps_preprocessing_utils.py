"""
Utilities for preprocessing raw RDPS data into the format required for inference.

Preprocessing pipeline:
  1. Load each variable from the raw RDPS file (native rotated-pole grid, 1076×1102).
  2. For anomaly variables, subtract the daily climatology (path_rdps_climatology)
     on the native grid before regridding.
  3. Regrid to the 8km North America ML grid (256×512) using precomputed xESMF
     nearest-neighbour sparse weights from path_grids.
  4. Normalize per variable specifications (rdps_variables.py).
  5. Load static fields (topography MF, land-sea mask sftlf) at 2km (1024×2048).
  6. Assemble and save the preprocessed NetCDF batch.

File naming conventions:
  Raw RDPS     : {path_rdps}/{YYYYMMDDCC}_{FFF}.nc
                   CC = run cycle (00), FFF = forecast hour (007…012)
  Climatology  : {path_rdps_climatology}/{var_name}/rdps_climatology_{var}_{MM-DD}T{HH}.nc
  Weights      : {path_grids}/regridding_weights/rdps_ec_ml_import_full-8km_north_america_ml/*.nc
  Grid         : {path_grids}/2km_north_america_ml.grid.nc  (output lat/lon coordinates)
  Static MF    : {path_hrdps_mf}          (single file, var: MF)
  Static sftlf : {path_hrdps_sftlf}       (single file, var: HRDPS_sftlf)
"""

import logging
import re
import xarray
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from resoterre.data_management.netcdf_utils import CFVariables
from resoterre.datasets.rdps.rdps_variables import rdps_variables as RDPS_VARIABLE_SPECS
from resoterre.ml.data_loader_utils import normalize

logger = logging.getLogger(__name__)


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
# Regridding (xESMF sparse weights)
# ---------------------------------------------------------------------------

# Flat size of the native RDPS rotated-pole grid (1076 rlat × 1102 rlon)
_RDPS_NATIVE_GRID_SIZE = 1076 * 1102


def load_regrid_weights(path_grids: Path) -> csr_matrix:
    """
    Load the precomputed xESMF nearest-neighbour sparse weights for regridding
    from the native RDPS grid (1076×1102) to the 8km North America ML grid (256×512).

    Parameters
    ----------
    path_grids : Path
        Root of the grids directory containing regridding_weights/.

    Returns
    -------
    csr_matrix
        Sparse weight matrix of shape (256*512, 1076*1102).
    """
    weights_dir = path_grids / "regridding_weights" / "rdps_ec_ml_import_full-8km_north_america_ml"
    weight_files = list(weights_dir.glob("*.nc"))
    if not weight_files:
        raise FileNotFoundError(f"No weight file found in {weights_dir}")
    ds = xarray.open_dataset(weight_files[0])
    row = ds["row"].values - 1  # 1-indexed -> 0-indexed
    col = ds["col"].values - 1
    S = ds["S"].values
    ds.close()
    n_target = row.size
    return csr_matrix((S, (row, col)), shape=(n_target, _RDPS_NATIVE_GRID_SIZE))


def apply_regrid_weights(
    field: np.ndarray, weights: csr_matrix, target_h: int, target_w: int
) -> np.ndarray:
    """
    Apply sparse regridding weights to a 2D (h, w) field.

    Parameters
    ----------
    field : np.ndarray
        2D array on the native RDPS grid (1076×1102).
    weights : csr_matrix
        Sparse weight matrix from load_regrid_weights().
    target_h, target_w : int
        Target spatial dimensions (e.g. 256, 512 for the 8km grid).

    Returns
    -------
    np.ndarray
        Regridded 2D array of shape (target_h, target_w).
    """
    regridded = weights @ field.ravel().astype(np.float64)
    return regridded.reshape(target_h, target_w).astype(np.float32)


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
        logger.warning("No normalization spec for '%s', using data min/max", lookup)
        return normalize(data, mode=(-1, 1))

    spec = RDPS_VARIABLE_SPECS[lookup]
    if spec.normalize_min is None or spec.normalize_max is None:
        logger.warning("normalize_min/max not set for '%s', using data min/max", lookup)
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
            raise ValueError(f"Topography file has unexpected shape {mf.shape}, expected ({h_out}, {w_out})")
        # Normalize: clip & scale elevation to [-1, 1]
        # Spec: range roughly [-5, 5000] m, normalize over [0, 4000]
        static[0] = normalize(mf, mode=(-1, 1), valid_min=0.0, valid_max=4000.0)
    else:
        logger.warning("path_hrdps_mf not found – topography channel set to zeros")

    # Channel 1 – land-sea mask (sftlf), already in [0, 1]
    if path_hrdps_sftlf is not None and Path(path_hrdps_sftlf).exists():
        ds = xarray.open_dataset(path_hrdps_sftlf, decode_timedelta=False)
        sftlf = ds["HRDPS_sftlf"].values.squeeze().astype(np.float32)  # (1024, 2048)
        ds.close()
        if sftlf.shape != (h_out, w_out):
            raise ValueError(f"Land-sea mask file has unexpected shape {sftlf.shape}, expected ({h_out}, {w_out})")
        # Normalize [0, 1] -> [-1, 1]
        static[1] = normalize(sftlf, mode=(-1, 1), valid_min=0.0, valid_max=1.0)
    else:
        logger.warning("path_hrdps_sftlf not found – land-sea mask channel set to zeros")

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
    path_grids: Path,
    h_in: int = 256,
    w_in: int = 512,
    h_out: int = 1024,
    w_out: int = 2048,
    path_rdps_climatology: Path | None = None,
    path_hrdps_mf: Path | None = None,
    path_hrdps_sftlf: Path | None = None,
) -> str:
    """
    Create a single preprocessed batch from one raw RDPS file.

    For each variable: extract from raw RDPS -> subtract climatology if anomaly
    -> regrid to 8km using xESMF sparse weights -> normalize.
    Then loads static fields and assembles the output NetCDF.

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
    path_grids : Path
        Root of the grids directory containing regridding_weights/ and *.grid.nc files.
    h_in, w_in : int
        Input spatial dimensions (8km grid: 256×512).
    h_out, w_out : int
        Output spatial dimensions (2km grid: 1024×2048).
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

    logger.info("File: %s | valid at %s | input (%d, %d) | output (%d, %d)",
                raw_rdps_file.name, valid_time, h_in, w_in, h_out, w_out)

    # Load regridding weights once for all variables
    weights = load_regrid_weights(path_grids)

    input_data = np.zeros((1, n_input_channels, h_in, w_in), dtype=np.float32)

    for i, var_name in enumerate(rdps_variables):
        is_anomaly = var_name in anomaly_variables

        # Load from raw RDPS on the native grid (1076×1102)
        field = load_from_raw_rdps(raw_rdps_file, var_name)

        # Subtract climatology before regridding (on native grid)
        if is_anomaly:
            clim = load_climatology_field(path_rdps_climatology, var_name, valid_time) \
                if path_rdps_climatology else None
            if clim is not None:
                field = field - clim
            else:
                logger.warning("No climatology for '%s' at %s", var_name, valid_time)

        # Regrid to 8km using xESMF sparse weights
        field = apply_regrid_weights(field, weights, h_in, w_in)

        # --- Normalize ---
        normalized = normalize_variable(field, var_name, is_anomaly=is_anomaly)
        input_data[0, i] = normalized

        tag = "anomaly" if is_anomaly else "raw"
        logger.debug("[%2d/%d] %s (%s): range=[%.3g, %.3g] -> norm=[%.3g, %.3g]",
                     i + 1, n_input_channels, var_name, tag,
                     field.min(), field.max(), normalized.min(), normalized.max())

    # --- Static fields ---
    logger.debug("Loading static fields...")
    static = load_static_fields(path_hrdps_mf, path_hrdps_sftlf, h_out, w_out)
    input_last_layer = static[np.newaxis, :, :, :]  # (1, 2, h_out, w_out)

    # --- Spatial / temporal coordinates ---
    dt = valid_time.astype("datetime64[s]").astype(object)

    # Read exact lat/lon from the 2km grid definition file
    grid_file = path_grids / "2km_north_america_ml.grid.nc"
    ds_grid = xarray.open_dataset(grid_file)
    lat_1d = ds_grid["lat"].values.astype(np.float32)
    lon_1d = ds_grid["lon"].values.astype(np.float32)
    ds_grid.close()

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

    ds_out = xarray.Dataset(
        data_vars=cf_vars,
        coords=cf_coords,
        attrs={"Conventions": "CF-1.6", "description": "Preprocessed RDPS data for downscaling inference"},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(output_path, engine="h5netcdf")

    return str(output_path)
