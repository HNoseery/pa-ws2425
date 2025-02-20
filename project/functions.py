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

            target = hdf_file[path]

            if attr_key in target.attrs:
                return target.attrs[attr_key]
            else:
                warnings.warn(f"Metadata attribute '{attr_key}' not found in '{path}'.", UserWarning)
                return None
    except KeyError:
        warnings.warn(f"Group or dataset '{path}' not found in the HDF5 file.", UserWarning)
        return None



def read_data(file: str, path: str) -> NDArray | None:
    import h5py
    import warnings
    """
        Reads a dataset from an HDF5 file and returns it as a 1D numpy array.

        Args:
            file (str): Path to the HDF5 file
            path (str): Path to the dataset within the HDF5 file

        Returns:
            NDArray | None: 1D numpy array if dataset exists, None otherwise

        Issues warnings for:
        - Non-existent paths
        - Paths pointing to groups instead of datasets
        """
    try:


      with h5py.File(file, 'r') as hdf_file:

            if path not in hdf_file:
                warnings.warn(f"Path '{path}' not found in HDF5 file.", UserWarning)
                return None

            obj = hdf_file[path]
            if not isinstance(obj, h5py.Dataset):
                warnings.warn(f"Path '{path}' points to a group, not a dataset.", UserWarning)
                return None


            data = np.array(obj)
            return data.squeeze()  # Remove singleton dimensions

    except Exception as e:
        warnings.warn(f"Error reading dataset '{path}': {str(e)}", UserWarning)
        return None


def check_equal_length(*arrays: NDArray) -> bool:
    """
       Checks if all input arrays have the same first dimension length.

       Args:
           *arrays (NDArray): Variable number of numpy arrays to compare

       Returns:
           bool: True if all arrays have the same length in their first dimension, False otherwise
       """
    if not arrays:
        return True

    first_length = arrays[0].shape[0]

    for arr in arrays[1:]:
        try:
            if arr.shape[0] != first_length:
                return False
        except IndexError:

            return False

    return True


def process_time_data(data: NDArray) -> NDArray:
    import numpy as np
    from numpy.typing import NDArray

    """
       Convert millisecond-precision timestamps to seconds since first measurement.

       Args:
           data: Array of timestamps in milliseconds (typically UNIX epoch format)

       Returns:
           NDArray: Time values in seconds relative to first measurement

       Example:
           >>> process_time_data(np.array([1640995200000, 1640995201000, 1640995202000]))
           array([0., 1., 2.])
       """
    if data.size == 0:
        return np.array([])


    relative_time = (data - data[0]) / 1000.0
    return relative_time.astype(np.float64)


def remove_negatives(array: NDArray) -> NDArray:
    """
       Replace all negative values in an array with NaN.

       Args:
           array: Input array containing numerical values

       Returns:
           NDArray: Processed array with negative values replaced by NaN

       Example:
           >>> remove_negatives(np.array([-1, 2, -3, 4]))
           array([nan,  2., nan,  4.])
       """
    if array.size == 0:
        return array


    processed = array.copy().astype(np.float64)  # Ensure float type for NaN support


    processed[processed < 0] = np.nan

    return processed


def linear_interpolation(
    time: NDArray, start_time: float, end_time: float, start_y: float, end_y: float
) -> NDArray:
    """
        Perform linear interpolation between two points (start_time, start_y) and (end_time, end_y).

        Args:
            time: Array of time values where interpolation should be calculated
            start_time: Time value at the start of the interval
            end_time: Time value at the end of the interval
            start_y: Y-value at start_time
            end_y: Y-value at end_time

        Returns:
            NDArray: Interpolated values at each time point

        Formula:
            y = start_y + (end_y - start_y) * (time - start_time) / (end_time - start_time)
        """

    time_diff = end_time - start_time
    ratio = (time - start_time) / time_diff


    interpolated = start_y + (end_y - start_y) * ratio

    return interpolated.astype(np.float64)


def interpolate_nan_data(time: NDArray, y_data: NDArray) -> NDArray:
    from typing import Optional
    """
        Replace NaN values in y_data using linear interpolation between valid data points.

        Args:
            time: Array of time values corresponding to measurements
            y_data: Array with measurement data containing NaNs

        Returns:
            NDArray: Processed array with NaNs replaced by interpolated values

        Raises:
            ValueError: If first/last value is NaN or unclosed gap exists
        """

    if np.isnan(y_data[0]) or np.isnan(y_data[-1]):
        raise ValueError("First or last value in y_data is NaN")

    interpolated_data = y_data.copy().astype(np.float64)  # Force float for NaN handling
    active_gap = False
    start_index: Optional[int] = None

    for i in range(len(interpolated_data)):
        current_val = interpolated_data[i]

        if np.isnan(current_val):
            if not active_gap:

                start_index = i
                active_gap = True
        else:
            if active_gap:

                end_index = i


                start_time = time[start_index - 1]
                end_time = time[end_index]
                start_y = interpolated_data[start_index - 1]
                end_y = interpolated_data[end_index]


                gap_time = time[start_index: end_index]


                interpolated_values = linear_interpolation(
                    gap_time, start_time, end_time, start_y, end_y
                )


                interpolated_data[start_index: end_index] = interpolated_values
                active_gap = False
                start_index = None


    if active_gap:
        raise ValueError("Unclosed gap detected - last value should be valid")

    return interpolated_data


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
    padded_data = np.pad(array=data, pad_width=pad_width, mode="edge")
    for i in range(pad_width, padded_data.size - pad_width):
        # Implementieren Sie hier den SMA!
        window = padded_data[i - pad_width: i + pad_width + 1]
        sma = np.mean(window)
        output.append(sma)
    return np.array(output)



def calc_heater_heat_flux(P_heater: float, eta_heater: float) -> float:
    """
       Calculate the effective heat flux from the heater accounting for efficiency.

       Args:
           P_heater: Nominal heater power [W]
           eta_heater: Heater efficiency factor [0-1]

       Returns:
           Effective heat flux Q_zu [W]

       Implements:
           Q_zu = P_heater * eta_heater
       """
    return P_heater * eta_heater


def calc_convective_heat_flow(
    k_tank: float, area_tank: float, t_total: float, t_env: float
) -> float:
    """
       Calculate convective heat loss from the tank to the environment.

       Args:
           k_tank (float): Heat transfer coefficient of the tank material [W/(m²·K)]
           area_tank (float): Outer surface area of the tank [m²]
           t_total (float): Current temperature of the tank (equals liquid temperature) [°C or K]
           t_env (float): Environmental temperature [°C or K]

       Returns:
           float: Convective heat loss Q_ab [W]

       Formula:
           Q_ab = k_tank * area_tank * (t_total - t_env)
       """
    return k_tank * area_tank * (t_total - t_env)


def calc_mass(
    level_data: NDArray, tank_footprint: float, density: float
) -> NDArray:
    """
       Calculate beer mass in tank over time from fill level measurements.

       Args:
           level_data: Array of fill heights [m]
           tank_footprint: Tank base area (A_b) [m²]
           density: Beer density (ρ) [kg/m³]

       Returns:
           NDArray: Mass values [kg] at each time step

       Formula:
           m(t) = ρ * A_b * h(t)
       """
    return density * tank_footprint * level_data


def calc_enthalpy(
    mass: float, specific_heat_capacity: float, temperature: float
) -> float:
    """
       Calculate enthalpy of beer in tank at a specific time.

       Args:
           mass: Mass of beer at time t* [kg]
           specific_heat_capacity: Specific heat capacity of beer [J/(kg·K)]
           temperature: Temperature of beer (T_zu) [K]

       Returns:
           float: Enthalpy H_zu [J]

       Formula:
           H_zu(t*) = m(t*) * c * T_zu
       """
    return mass * specific_heat_capacity * temperature


def store_plot_data(
    data: dict[str, NDArray], file_path: str, group_path: str, metadata: dict[str, Any]
) -> None:
    import pandas as pd
    from typing import Any, Tuple

    """
       Store processed data and metadata in an HDF5 file using pandas.HDFStore.

       Args:
           data: Dictionary of data arrays to store (keys = column names)
           file_path: Path to the output HDF5 file
           group_path: Group path within the HDF5 file
           metadata: Dictionary of metadata attributes for the group
       """
    # Convert data dictionary to DataFrame
    df = pd.DataFrame(data)

    # Create HDF5 file and store data with metadata
    with pd.HDFStore(file_path, mode='a') as store:
        # Store DataFrame (overwrite if exists)
        store.put(
            key=group_path,
            value=df,
            format='table',  # Required for attributes
            append=False
        )

        # Add metadata as group attributes
        storer = store.get_storer(group_path)
        for key, value in metadata.items():
            storer.attrs[key] = value


def read_plot_data(
    file_path: str, group_path: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import pandas as pd
    from typing import Any, Tuple

    """
       Read archived data and metadata from HDF5 file for plotting.

       Args:
           file: Path to HDF5 file
           group_path: Group path containing the dataset

       Returns:
           Tuple containing:
           - pd.DataFrame: Archived data
           - dict: Plot labels metadata (legend title, axis labels with units)
       """
    with pd.HDFStore(file_path, mode='r') as store:
        # Read DataFrame
        df = store.get(group_path)

        # Read metadata attributes
        metadata = {}
        if store.get_storer(group_path):
            storer = store.get_storer(group_path)
            metadata = {key: storer.attrs[key] for key in storer.attrs._v_attrnames}  # Fixed

        # Build plot labels
        plot_labels = {
            "legend_title": metadata.get("legend_title", "Brewing Process"),
            "x_label": f"{metadata.get('x_label', 'Time')} [{metadata.get('x_unit', 's')}]",
            "y_label": f"{metadata.get('y_label', 'Inner Energy')} [{metadata.get('y_unit', 'J')}]"
        }

    return df, plot_labels


def plot_data(data: pd.DataFrame, formats: dict[str, str]) -> Figure:
    """
        Plot inner energy data in gigajoules over time in hours.

        Args:
            data: DataFrame containing time and inner energy data
            formats: Dictionary with plot formatting information

        Returns:
            matplotlib.figure.Figure: The generated plot figure
        """
    fig = plt.figure(figsize=(12, 6))

    # Convert units
    time_hours = data['time'] / 3600  # Seconds → hours
    x_label = f"{formats['x_label'].split(' [')[0]} [hours]"
    y_label = f"Inner Energy [GJ]"

    # Plot all energy curves
    energy_cols = [col for col in data.columns if col.startswith('inner_energy_k_')]
    for col in energy_cols:
        filter_size = col.split('_')[-1]
        plt.plot(
            time_hours,
            data[col] / 1e9,  # Convert J → GJ
            label=f'Filter size {filter_size}',
            linewidth=1
        )

    # Style plot
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(formats['legend_title'], fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Filter Sizes', title_fontsize=12)
    plt.tight_layout()

    return fig


def publish_plot(
    fig: Figure, source_paths: str | list[str], destination_path: str
) -> None:


    import os
    from datetime import datetime

    """
        Save and publish a plot with institutional tagging requirements.

        Args:
            fig: Matplotlib figure to publish
            source_paths: Path(s) to source data files
            destination_path: Output directory for published materials
        """

    os.makedirs(destination_path, exist_ok=True)

    # Generate unique ID
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    matrikelnummer = 2592668
    plot_id = f"GdD_WS_2425_{matrikelnummer}_{current_time}"

    # Tag the plot and get the PlotIDTransfer object from tagplot()
    id_transfer = tagplot(  # <-- tagplot() returns PlotIDTransfer
        figs=[fig],
        ids=[plot_id],
        engine="matplotlib",
        save_dir=destination_path
    )

    # Publish the data using the PlotIDTransfer object
    publish(
        figs_and_ids=id_transfer,  # <-- Pass the object returned by tagplot()
        src_datapath=source_paths,
        dst_path=destination_path
    )


if __name__ == "__main__":
    pass
