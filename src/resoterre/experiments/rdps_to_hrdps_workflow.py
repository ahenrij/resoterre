"""Workflow utilities for RDPS to HRDPS downscaling."""

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray

from resoterre.config_utils import config_from_yaml, known_configs
from resoterre.data_management.netcdf_utils import CFVariables
from resoterre.datasets.hrdps.hrdps_variables import hrdps_variables
from resoterre.datasets.rdps.rdps_preprocessing_utils import create_preprocessed_batch
from resoterre.hybrid_data_loaders.rdps_to_hrdps import post_process_model_output
from resoterre.logging_utils import start_root_logger
from resoterre.ml.network_manager import NeuralNetworksManager, NeuralNetworksManagerConfig
from resoterre.ml.neural_networks_unet import UNet
from resoterre.utils import TemplateStore


@dataclass(frozen=True, slots=True)
class RDPSToHRDPSOnDiskConfig:
    """
    Configuration for the RDPS to HRDPS data workflow.

    Attributes
    ----------
    path_workflow : Path
        Path to the main workflow directory.
    path_logs : Path, optional
        Path to the logs directory.
    path_figures : Path, optional
        Path to the figures directory.
    path_rdps_regrid : Path, optional
        Path to the RDPS regridded data directory.
    path_hrdps_regrid : Path, optional
        Path to the HRDPS regridded data directory.
    path_ml_data : Path, optional
        Path for the output machine learning data.
    path_rdps : Path, optional
        Path to the raw RDPS data directory.
    path_rdps_climatology : Path, optional
        Path to the RDPS climatology data directory.
    path_hrdps : Path, optional
        Path to the raw HRDPS data directory.
    path_hrdps_climatology : Path, optional
        Path to the HRDPS climatology data directory.
    path_hrdps_mask : Path, optional
        Path to the HRDPS mask file.
    path_hrdps_mf : Path, optional
        Path to the HRDPS topography file.
    path_hrdps_sftlf : Path, optional
        Path to the HRDPS land-sea mask file.
    path_grids : Path, optional
        Path to the grids directory.
    random_seed : int, optional
        Random seed for reproducibility.
    rdps_input_validation_batch_size : int
        Batch size for RDPS input validation.
    data_loader_snakemake_batch_size : int
        How many time steps are processed in a single snakemake job. This should be a multiple of save_batch_size.
    grid_input_for_ml : str, optional
        Grid name for input data in machine learning.
    grid_output_for_ml : str, optional
        Grid name for output data in machine learning.
    start_datetime : datetime.datetime, optional
        Start datetime for data processing.
    end_datetime : datetime.datetime, optional
        End datetime for data processing.
    rdps_variables : list[str]
        List of RDPS variable names to process.
    hrdps_variables : list[str]
        List of HRDPS variable names to process.
    forecast_hours : list[int]
        List of forecast hours to include.
        Only tested with [7, 8, 9, 10, 11, 12].
    rdps_window_size : int, optional
        Size of the RDPS patches (if using tiling).
    overlap_factor : int, optional
        Overlap factor for RDPS patches.
    hrdps_required_unmasked_fraction : float, optional
        Required unmasked fraction for HRDPS data.
    temporal_window : int, optional
        Temporal window size for including context (1 = one timestep on each side of current timestep).
    variables_with_temporal_context : list[str]
        List of variable names that should include temporal context.
    anomaly_variables : list[str]
        List of variable names to be treated as anomalies.
    normalize : bool
        Whether to normalize the data.
    train_fraction : float
        Fraction of data to use for training.
    validation_fraction : float
        Fraction of data to use for validation.
    test_fraction : float
        Fraction of data to use for testing.
    save_batch_size : int
        Batch size for saving data.
    """

    path_workflow: Path
    path_logs: Path | None = None
    path_figures: Path | None = None
    path_rdps_regrid: Path | None = None
    path_hrdps_regrid: Path | None = None
    path_ml_data: Path | None = None
    path_rdps: Path | None = None
    path_rdps_climatology: Path | None = None
    path_hrdps: Path | None = None
    path_hrdps_climatology: Path | None = None
    path_hrdps_mask: Path | None = None
    path_hrdps_mf: Path | None = None
    path_hrdps_sftlf: Path | None = None
    path_grids: Path | None = None
    random_seed: int | None = 0
    rdps_input_validation_batch_size: int = 32  # ToDo: this is also used for hrdps validation batch size
    data_loader_snakemake_batch_size: int = 128
    # ToDo: many more workflow settings are needed here
    grid_input_for_ml: str | None = None
    grid_output_for_ml: str | None = None
    start_datetime: datetime.datetime | None = None
    end_datetime: datetime.datetime | None = None
    rdps_variables: list[str] = field(default_factory=list)
    hrdps_variables: list[str] = field(default_factory=list)
    forecast_hours: list[int] = field(default_factory=list)
    rdps_window_size: int | None = None
    overlap_factor: int | None = None  # preferably a divisor of rdps_window_size
    hrdps_required_unmasked_fraction: float | None = None
    temporal_window: int | None = None
    variables_with_temporal_context: list[str] = field(default_factory=list)
    anomaly_variables: list[str] = field(default_factory=list)
    normalize: bool = True
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    save_batch_size: int = 1


@dataclass(frozen=True, slots=True)
class RDPSToHRDPSPreprocessingConfig:
    """
    Configuration for RDPS to HRDPS preprocessing.

    Attributes
    ----------
    path_logs : Path, optional
        Path to the logs directory.
    path_output : Path, optional
        Path to the output directory where preprocessed data will be saved.
    path_rdps : Path, optional
        Path to the raw RDPS data directory.
    path_rdps_regrid : Path, optional
        Path to the pre-regridded RDPS variable files (production pipeline).
    path_hrdps : Path, optional
        Path to the raw HRDPS data directory (for validation/target data).
    path_rdps_climatology : Path, optional
        Path to the RDPS climatology data directory.
    path_hrdps_climatology : Path, optional
        Path to the HRDPS climatology data directory.
    path_hrdps_mask : Path, optional
        Path to the HRDPS mask file.
    path_hrdps_mf : Path, optional
        Path to the HRDPS topography file.
    path_hrdps_sftlf : Path, optional
        Path to the HRDPS land-sea mask file.
    path_grids : Path, optional
        Path to the grids directory.
    grid_input_for_ml : str, optional
        Grid name for input data in machine learning.
    grid_output_for_ml : str, optional
        Grid name for output data in machine learning.
    start_datetime : datetime.datetime, optional
        Start datetime for data processing.
    end_datetime : datetime.datetime, optional
        End datetime for data processing.
    rdps_variables : list[str]
        List of RDPS variable names to process.
    hrdps_variables : list[str]
        List of HRDPS variable names to process.
    normalize : bool
        Whether to normalize the data.
    save_batch_size : int
        Batch size for saving preprocessed data.
    anomaly_variables : list[str]
        List of variable names to be treated as anomalies.
    temporal_window : int, optional
        Temporal window size for including context.
    variables_with_temporal_context : list[str]
        List of variable names that should include temporal context.
    """

    path_logs: Path | None = None
    path_output: Path | None = None
    path_rdps: Path | None = None
    path_rdps_regrid: Path | None = None
    path_hrdps: Path | None = None
    path_rdps_climatology: Path | None = None
    path_hrdps_climatology: Path | None = None
    path_hrdps_mask: Path | None = None
    path_hrdps_mf: Path | None = None
    path_hrdps_sftlf: Path | None = None
    path_grids: Path | None = None
    grid_input_for_ml: str | None = None
    grid_output_for_ml: str | None = None
    start_datetime: datetime.datetime | None = None
    end_datetime: datetime.datetime | None = None
    rdps_variables: list[str] = field(default_factory=list)
    hrdps_variables: list[str] = field(default_factory=list)
    normalize: bool = True
    save_batch_size: int = 1
    anomaly_variables: list[str] = field(default_factory=list)
    temporal_window: int | None = None
    variables_with_temporal_context: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RDPSToHRDPSInferenceConfig:
    """
    Configuration for RDPS to HRDPS inference.

    Attributes
    ----------
    path_logs : Path, optional
        Path to the logs directory.
    path_models : Path, optional
        Path to the directory containing the trained models.
    path_output : Path, optional
        Path to the output directory where results will be saved.
    path_preprocessed_batch : Path, optional
        Path to the preprocessed data batch for inference.
    search_path: Path, optional
        Path to the directory containing preprocessed data files for inference.
    glob_pattern: str, optional
        Glob pattern to match multiple preprocessed data files for inference.
    experiment_name : str
        Name of the experiment.
    device : str
        Device to use for inference (e.g., "cpu" or "cuda").
    """

    path_logs: Path | None = None
    path_models: Path | None = None
    path_output: Path | None = None
    path_preprocessed_batch: Path | None = None
    search_path: Path | None = None
    glob_pattern: str | None = None
    experiment_name: str = "test"
    device: str = "cpu"


def preprocessing_raw_to_preprocessed(
    config: RDPSToHRDPSPreprocessingConfig | dict[str, Any] | Path | str,
) -> list[str]:
    """
    Workflow for preprocessing raw RDPS data into the format required for inference.

    This function converts raw RDPS meteorological data files into preprocessed NetCDF batches
    that can be directly used by the inference pipeline.

    Parameters
    ----------
    config : RDPSToHRDPSPreprocessingConfig | dict[str, Any] | Path | str
        Configuration for the preprocessing process, including as a dictionary or a path to a YAML file.

    Returns
    -------
    list[str]
        List of paths to the saved preprocessed batch files.

    Notes
    -----
    The preprocessing steps include:
    1. Load raw RDPS data files
    2. Regrid to the target ML grid resolution
    3. Apply normalization and anomaly processing
    4. Create input tensors with temporal context if needed
    5. Save as NetCDF batches in the expected format

    Expected output format matches what inference expects:
    - input_first_block: (n_samples, in_channels, h_in, w_in)
    - input_last_layer: (n_samples, last_layer_channels, h_out, w_out)
    - target (optional): (n_samples, target_channels, h_out, w_out)
    - Coordinates: year, month, day, hour, lat, lon, etc.
    """
    if isinstance(config, RDPSToHRDPSPreprocessingConfig):
        config_obj = config
    else:
        config_obj = config_from_yaml(
            RDPSToHRDPSPreprocessingConfig, config, known_custom_config_dict=known_configs
        )

    if config_obj.path_rdps is None:
        raise ValueError("path_rdps must be specified in the config for raw RDPS data.")
    if config_obj.path_output is None:
        raise ValueError("path_output must be specified in the config.")

    templates = TemplateStore({"log_file": "${path_logs}/${timestamp}_preprocessing.log"})
    templates.add_substitutes(path_logs=str(config_obj.path_logs))
    if config_obj.path_logs is not None:
        _ = start_root_logger(
            templates=templates,
            disable_loggers=[
                "numba.core.byteflow",
                "numba.core.ssa",
                "numba.core.interpreter",
                "matplotlib.font_manager",
                "matplotlib.colorbar",
            ],
        )

    print("\n" + "=" * 80)
    print("PREPROCESSING PIPELINE - Raw RDPS to Preprocessed Format")
    print("=" * 80 + "\n")
    print(f"Input RDPS data path  : {config_obj.path_rdps}")
    print(f"Pre-regridded path    : {config_obj.path_rdps_regrid}")
    print(f"Climatology path      : {config_obj.path_rdps_climatology}")
    print(f"Topography file (MF)  : {config_obj.path_hrdps_mf}")
    print(f"Land-sea mask (sftlf) : {config_obj.path_hrdps_sftlf}")
    print(f"Output path           : {config_obj.path_output}")
    print(f"RDPS variables        : {config_obj.rdps_variables}")
    print(f"HRDPS variables       : {config_obj.hrdps_variables}")
    print(f"Anomaly variables     : {config_obj.anomaly_variables}")

    # Resolve target grid sizes from config grid names
    grid_sizes = {
        "8km_north_america_ml": (256, 512),
        "2km_north_america_ml": (1024, 2048),
    }
    h_in, w_in = grid_sizes.get(config_obj.grid_input_for_ml, (256, 512))
    h_out, w_out = grid_sizes.get(config_obj.grid_output_for_ml, (1024, 2048))

    # Scan raw RDPS files
    rdps_path = Path(config_obj.path_rdps)
    rdps_files = sorted(rdps_path.glob("*.nc"))
    if not rdps_files:
        raise FileNotFoundError(f"No NetCDF files found in {rdps_path}")

    print(f"\nFound {len(rdps_files)} raw RDPS file(s)")

    output_dir = Path(config_obj.path_output)
    saved_files = []

    for batch_idx, raw_file in enumerate(rdps_files):
        print(f"\n[{batch_idx + 1}/{len(rdps_files)}] Processing {raw_file.name} ...")
        output_path = output_dir / f"preprocessed_{raw_file.stem}.nc"

        saved_file = create_preprocessed_batch(
            raw_rdps_file=raw_file,
            rdps_variables=config_obj.rdps_variables,
            hrdps_variables=config_obj.hrdps_variables,
            anomaly_variables=config_obj.anomaly_variables,
            output_path=output_path,
            path_grids=Path(config_obj.path_grids),
            h_in=h_in,
            w_in=w_in,
            h_out=h_out,
            w_out=w_out,
            path_rdps_climatology=Path(config_obj.path_rdps_climatology) if config_obj.path_rdps_climatology else None,
            path_hrdps_mf=Path(config_obj.path_hrdps_mf) if config_obj.path_hrdps_mf else None,
            path_hrdps_sftlf=Path(config_obj.path_hrdps_sftlf) if config_obj.path_hrdps_sftlf else None,
        )
        print(f"  Saved: {saved_file}")
        saved_files.append(saved_file)

    print(f"\nPreprocessing complete! {len(saved_files)} file(s) written to {output_dir}")
    print(f"\nNext step, run inference:")
    print(f"  python scripts/inference/downscaling_inference_rdps_to_hrdps.py <config>")

    return saved_files


def save_model_output(
    config: RDPSToHRDPSInferenceConfig, data_sample: xarray.Dataset, output_variables: dict[str, np.ndarray]
) -> list[str]:
    """
    Save the model output to NetCDF files.

    Parameters
    ----------
    config : RDPSToHRDPSInferenceConfig
        Configuration for the inference process.
    data_sample : xarray.Dataset
        The input data sample used for inference.
    output_variables : dict[str, np.ndarray]
        The output variables from the model, keyed by variable name.

    Returns
    -------
    list[str]
        List of paths to the saved output files.
    """
    if config.path_output is None:
        raise ValueError("config.path_output must be specified to save model output.")
    list_of_saved_files = []
    for i in range(data_sample.sizes["sample"]):
        for j in range(data_sample.sizes["target_channel"]):
            variable_name = str(data_sample["output_variables"].values[j])
            data = {}
            year = int(data_sample["year"].values[i])
            month = int(data_sample["month"].values[i])
            day = int(data_sample["day"].values[i])
            hour = int(data_sample["hour"].values[i])
            for key in ["height_out_idx", "width_out_idx", "lat", "lon"]:
                data[key] = data_sample[key].values[i, :]

            cf_coordinates = CFVariables()
            for key in ["lat", "lon", "height_out_idx", "width_out_idx"]:
                if key in ["lat", "height_out_idx"]:
                    dims = ("lat",)
                elif key in ["lon", "width_out_idx"]:
                    dims = ("lon",)
                else:
                    raise ValueError(f"Unexpected coordinate key: {key}")
                cf_coordinates.add(
                    key,
                    dims=dims,
                    data=data[key],
                    dtype=data_sample[key].dtype,
                    attributes=data_sample[key].attrs,
                )
            datetime_str = f"{year}-{month:02d}-{day:02d}T{hour:02d}:00:00"
            cf_coordinates.add("time", dims=(), data=np.array([datetime_str], dtype="datetime64[ns]"))
            cf_variables = CFVariables()
            # ToDo: add standard_name and long_name mapping to hrdps_variables and add it to attributes
            #       careful, this is post accumulation to quantity
            cf_variables.add(
                variable_name,
                dims=("lat", "lon"),
                attributes={"units": hrdps_variables[variable_name].units},
                data=output_variables[variable_name][i, 0, :, :],  # channel index was already extracted
                dtype=np.float32,
                zlib=True,
                complevel=4,
            )
            cf_attrs = {"Conventions": "CF-1.6"}
            ds_out = xarray.Dataset(data_vars=cf_variables, coords=cf_coordinates, attrs=cf_attrs)
            encoding = {"time": {"dtype": "int64", "calendar": "standard", "units": "hours since 1970-01-01 00:00:00"}}

            # ToDo: different tiles would have the same name, need to add tile indices to the path
            date_str = f"{year}{month:02d}{day:02d}{hour:02d}"
            save_file = f"{variable_name}/{date_str}_{data['height_out_idx'][0]}_{data['width_out_idx'][0]}.nc"
            path_output = Path(config.path_output, save_file)
            path_output.parent.mkdir(parents=True, exist_ok=True)
            ds_out.to_netcdf(path_output, engine="h5netcdf", encoding=encoding)
            list_of_saved_files.append(str(path_output))
    return list_of_saved_files


def inference_from_preprocessed_data(config: RDPSToHRDPSInferenceConfig | dict[str, Any] | Path | str) -> list[str]:
    """
    Workflow for performing inference from preprocessed RDPS data to HRDPS data.

    Parameters
    ----------
    config : RDPSToHRDPSInferenceConfig | dict[str, Any] | Path | str
        Configuration for the inference process, including as a dictionary or a path to a YAML file.

    Returns
    -------
    list[str]
        List of paths to the saved output files.
    """
    if isinstance(config, RDPSToHRDPSInferenceConfig):
        config_obj = config
    else:
        config_obj = config_from_yaml(RDPSToHRDPSInferenceConfig, config, known_custom_config_dict=known_configs)

    if config_obj.path_models is None:
        raise ValueError("path_models must be specified in the config.")
    if config_obj.path_output is None:
        raise ValueError("path_output must be specified in the config.")
    if config_obj.path_preprocessed_batch is None:
        raise NotImplementedError("Inference from something other than a preprocessed batch is not implemented.")

    templates = TemplateStore({"log_file": "${path_logs}/${timestamp}_inference.log"})
    templates.add_substitutes(path_logs=str(config_obj.path_logs))
    if config_obj.path_logs is not None:
        _ = start_root_logger(
            templates=templates,
            disable_loggers=[
                "numba.core.byteflow",
                "numba.core.ssa",
                "numba.core.interpreter",
                "matplotlib.font_manager",
                "matplotlib.colorbar",
            ],
        )

    # Load UNet
    network_manager, _ = NeuralNetworksManager.from_path_models(
        path_models=config_obj.path_models,
        neural_networks_classes={"UNet": UNet},
        experiment_name=config_obj.experiment_name,
        config=NeuralNetworksManagerConfig(device=config_obj.device),
    )

    # Load preprocessed data
    data_sample = xarray.open_dataset(config_obj.path_preprocessed_batch)
    x = torch.tensor(data_sample["input_first_block"].values).to(config_obj.device)
    x_last_layer = torch.tensor(data_sample["input_last_layer"].values).to(config_obj.device)
    output = network_manager.networks["UNet"](x=x, x_linear=None, x_last_layer=x_last_layer)

    # Denormalize and revert climatology removal
    output_variables = post_process_model_output(
        output, variable_names=list(map(str, data_sample["output_variables"].values))
    )

    # Save output
    list_of_saved_files = save_model_output(config_obj, data_sample, output_variables)

    return list_of_saved_files
