from typing import Any

import h5py as h5
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from plotid.publish import publish
from plotid.tagplot import tagplot

def read_metadata(file: str, path: str, attr_key: str) -> Any | None:
    import h5py
    import warnings

    """
       Reads metadata from an HDF5 file.

       Args:
           file (str): Path to the HDF5 file.
           path (str): Path to the HDF5 group or dataset within the file.
           attr_key (str): Name of the metadata attribute to read.

       Returns:
           any | None: The value of the metadata attribute if found, otherwise None.

       This function attempts to read the specified metadata attribute from the given HDF5 group or dataset.
       If the attribute, group, or dataset does not exist, a warning is issued, and the function returns None.
       """
    try:
        with h5py.File(file, 'r') as hdf_file:
            # Access the group or dataset
            target = hdf_file[path]
            # Read the metadata attribute
            if attr_key in target.attrs:
                return target.attrs[attr_key]
            else:
                warnings.warn(f"Metadata attribute '{attr_key}' not found in '{path}'.", UserWarning)
                return None
    except KeyError:
        warnings.warn(f"Group or dataset '{path}' not found in the HDF5 file.", UserWarning)
        return None



def read_data(file: str, path: str) -> NDArray | None:
    pass


def check_equal_length(*arrays: NDArray) -> bool:
    pass


def process_time_data(data: NDArray) -> NDArray:
    pass


def remove_negatives(array: NDArray) -> NDArray:
    pass


def linear_interpolation(
    time: NDArray, start_time: float, end_time: float, start_y: float, end_y: float
) -> NDArray:
    pass


def interpolate_nan_data(time: NDArray, y_data: NDArray) -> NDArray:
    pass


def filter_data(data: NDArray, window_size: int) -> NDArray:
    """Filter data using a moving average approach.

    Args:
        data (NDArray): Data to be filtered
        window_size (int): Window size of the filter

    Returns:
        NDArray: Filtered data
    """
    output = []
    pad_width = window_size // 2
    padded_data = np.pad(array=data, pad_width=pad_width, mode="empty")
    for i in range(pad_width, padded_data.size - pad_width):
        # Implementieren Sie hier den SMA!
        sma = []
        output.append(sma)
    return np.array(output)



def calc_heater_heat_flux(P_heater: float, eta_heater: float) -> float:
    pass


def calc_convective_heat_flow(
    k_tank: float, area_tank: float, t_total: float, t_env: float
) -> float:
    pass


def calc_mass_flow(
    level_data: NDArray, tank_footprint: float, density: float
) -> NDArray:
    pass


def calc_transported_power(
    mass_flow: float, specific_heat_capacity: float, temperature: float
) -> float:
    pass


def store_plot_data(
    data: dict[str, NDArray], file_path: str, group_path: str, metadata: dict[str, Any]
) -> None:
    pass


def read_plot_data(
    file_path: str, group_path: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pass


def plot_data(data: pd.DataFrame, formats: dict[str, str]) -> Figure:
    pass


def publish_plot(
    fig: Figure, source_paths: str | list[str], destination_path: str
) -> None:
    pass


if __name__ == "__main__":
    pass
