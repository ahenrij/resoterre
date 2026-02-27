# RDPS Preprocessing

This directory contains the preprocessing pipeline that converts raw RDPS forecast files into preprocessed NetCDF batches ready for inference.

---

## Overview

The pipeline reads raw RDPS files (`{YYYYMMDDCC}_{FFF}.nc`), normalizes the input variables, loads static fields (topography, land-sea mask), and writes one preprocessed NetCDF batch per input file.

Each output batch contains:
- `input_first_block` `(1, 10, 256, 512)`: normalized RDPS variables at 8 km
- `input_last_layer` `(1, 2, 1024, 2048)`: topography (MF) + land-sea mask at 2 km

Output files are named `preprocessed_{source_file_stem}.nc` (e.g., `preprocessed_2024010100_007.nc`).

---

## Configuration

Copy `configs/downscaling/downscaling_preprocessing_rdps_to_hrdps.yaml` and fill in the paths for your environment. The key paths are:

| Key | Description |
|-----|-------------|
| `path_rdps` | Directory of raw RDPS forecast files (`{YYYYMMDDCC}_{FFF}.nc`) |
| `path_rdps_regrid` | Pre-regridded variable files at 256×512 (fast path; optional but recommended) |
| `path_rdps_climatology` | Climatology files used to compute anomalies for `PN`, `GZ500`, `UU850`, `VV850` |
| `path_hrdps_mf` | Path to the topography file (`HRDPS_MF_2km_north_america_ml.nc`) |
| `path_hrdps_sftlf` | Path to the land-sea mask file (`HRDPS_sftlf_2km_north_america_ml.nc`) |
| `path_grids` | Directory containing grid definition files and regridding weights |
| `path_output` | Output directory for preprocessed NetCDF batches |
| `path_logs` | Directory for log files |

---

## Running Locally

```bash
python scripts/preprocessing/downscaling_preprocessing_rdps_to_hrdps.py \
  configs/downscaling/downscaling_preprocessing_rdps_to_hrdps.yaml
```

Add `--verbose` for debug-level logging.

---

## Running with Docker

### 1. Build the base image (once)

```bash
docker build -f docker/Dockerfile.base -t resoterre-base:latest .
```

### 2. Build the preprocessing image

```bash
docker build -f docker/Dockerfile.preprocessing -t resoterre-preprocessing:latest .
```

### 3. Run preprocessing

Mount the directories for your input data, static fields, and output. The container expects:

| Mount point | Content |
|-------------|---------|
| `/app/configs` | Directory containing your YAML config |
| `/data/rdps` | Raw RDPS forecast files |
| `/data/rdps_regrid` | Pre-regridded RDPS files (optional fast path) |
| `/data/rdps_climatology` | Climatology files |
| `/data/hrdps_regrid` | Static fields (MF + sftlf files) |
| `/data/grids` | Grid definition files |
| `/app/outputs` | Output directory for preprocessed batches |
| `/tmp/logs` | Log output |

Make sure the paths in your config match the mount points above, then run:

```bash
docker run --rm \
  -v $(pwd)/configs:/app/configs:ro \
  -v /path/to/RDPS:/data/rdps:ro \
  -v /path/to/RDPS_regrid:/data/rdps_regrid:ro \
  -v /path/to/RDPS_climatology:/data/rdps_climatology:ro \
  -v /path/to/HRDPS_regrid:/data/hrdps_regrid:ro \
  -v /path/to/grids:/data/grids:ro \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/tmp/logs \
  resoterre-preprocessing:latest
```

To override the default config path:

```bash
docker run --rm \
  ... \
  resoterre-preprocessing:latest /app/configs/my_custom_config.yaml
```
