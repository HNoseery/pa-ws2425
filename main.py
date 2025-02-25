import datetime
import os

import numpy as np

import project.functions as fn
from project.functions import read_metadata


import pandas as pd

from numpy.typing import NDArray
from typing import Any


def main():

    file_path = "/Users/HNoseery/Desktop/pa-ws2425/project/data/data_GdD_Datensatz_WS2425.h5"
    brewing = "brewing_0001"
    tank_id = "B002"

    measured_quantities = ("level", "temperature", "timestamp")

    raw_data = {}

    for quantity in measured_quantities:
        from project.functions import read_data

        dataset_path = f"{brewing}/{tank_id}/{quantity}"

        data = read_data(file_path,dataset_path)

        raw_data[quantity] = data

    for quantity, data in raw_data.items():
        from project.functions import check_equal_length

        if data is None:
            raise ValueError(f"Failed to read dataset: {quantity}")

        # Check if all datasets have equal length
    if not check_equal_length(*raw_data.values()):

        dataset_lengths = {k: len(v) for k, v in raw_data.items()}
        raise ValueError(
            f"Dataset length mismatch: {dataset_lengths}\n"
            "All measurement datasets must have the same length."
        )

    print("\nRaw Data Structure:")
    for key, value in raw_data.items():
        print(f"{key}:", "Data loaded" if value is not None else "No data")

    print("\nData Validation Successful!")
    print(f"All datasets have length: {len(next(iter(raw_data.values())))}")



    brewing_group_path = "brewing_0001"


    T_env = read_metadata(file_path, brewing_group_path, "T_env")

    specific_heat_capacity_beer = read_metadata(file_path,brewing_group_path,"specific_heat_capacity_beer")

    density_beer = read_metadata(file_path,brewing_group_path,"density_beer")



    tank_group_path = "brewing_0001/B002"

    mass_tank = read_metadata(file_path,tank_group_path,"mass_tank")


    surface_area_tank = read_metadata(file_path,tank_group_path,"surface_area_tank")


    footprint_tank = read_metadata(file_path,tank_group_path,"footprint_tank")


    heat_transfer_coeff_tank = read_metadata(file_path,tank_group_path,"heat_transfer_coeff_tank")


    specific_heat_capacity_tank = read_metadata(file_path,tank_group_path,"specific_heat_capacity_tank")


    power_heater = read_metadata(file_path,tank_group_path,"power_heater")


    efficiency_heater = read_metadata(file_path,tank_group_path,"efficiency_heater")

    df_data = {}


    from project.functions import process_time_data
    from project.functions import remove_negatives
    from project.functions import interpolate_nan_data
    from project.functions import filter_data
    from project.functions import calc_mass
    from project.functions import calc_enthalpy
    from project.functions import calc_heater_heat_flux
    from project.functions import calc_convective_heat_flow

    processed_data = {}
    df_data = {}


    filter_sizes = (4, 29, 45, 201)


    if 'timestamp' in raw_data and raw_data['timestamp'] is not None:
        df_data['time'] = process_time_data(raw_data['timestamp'])
    else:
        raise ValueError("Timestamp data missing - cannot process time data")


    for k in filter_sizes:

        temp_key = f"temperature_k_{k}"
        if 'temperature' in raw_data and raw_data['temperature'] is not None:
            processed_data[temp_key] = filter_data(
                data=raw_data['temperature'],
                window_size=k
            )


        level_key = f"level_k_{k}"
        if 'level' in raw_data and raw_data['level'] is not None:

            cleaned_level = remove_negatives(raw_data['level'])


            interpolated_level = interpolate_nan_data(
                time=df_data['time'],
                y_data=cleaned_level
            )


            processed_data[level_key] = filter_data(
                data=interpolated_level,
                window_size=k
            )
        temp_key = f"temperature_k_{k}"
        level_key = f"level_k_{k}"
        filtered_temp = processed_data[temp_key]
        filtered_level = processed_data[level_key]

        # 1. Calculate mass over time
        mass_data = calc_mass(
            level_data=filtered_level,
            tank_footprint=footprint_tank,
            density=density_beer
        )

        # 2. Calculate initial energy E0
        E0 = calc_enthalpy(
            mass=mass_tank,
            specific_heat_capacity=specific_heat_capacity_tank,
            temperature=T_env
        )

        # 3. Initialize energy storage
        inner_energy = []

        # 4. Get metadata parameters
        Q_zu = calc_heater_heat_flux(
            P_heater=power_heater,  # Replace with your actual variable name
            eta_heater=efficiency_heater
        )

        # 5. Time iteration loop
        for i, (current_time, current_temp) in enumerate(zip(df_data['time'], filtered_temp)):
            # Calculate convective heat loss
            Q_ab = calc_convective_heat_flow(
                k_tank=heat_transfer_coeff_tank,
                area_tank=surface_area_tank,
                t_total=current_temp,
                t_env=T_env
            )

            # Calculate current enthalpy
            H_zu = calc_enthalpy(
                mass = mass_data[i],
                specific_heat_capacity=specific_heat_capacity_beer,
                temperature=current_temp
            )

            # Calculate and store inner energy
            current_E = Q_zu - Q_ab + H_zu + E0
            inner_energy.append(current_E)

        # 6. Store results
        df_data[f"inner_energy_k_{k}"] = np.array(inner_energy)

    print("\nProcessed Data Structure:")
    for key in processed_data:
        print(f"- {key}: {len(processed_data[key])} samples")

    print("\nDF Data Structure:")
    for key in df_data:
        print(f"- {key}: {len(df_data[key])} samples")

        # ================== Data Archiving - 5a ==================
        # 1. Declare archive path and group
    h5_path = "/Users/HNoseery/Desktop/pa-ws2425/project/results/data_Gdb_plot_WS2425.h5"
    group_path = f"/brewing_{brewing}/tank_{tank_id}"

    # 2. Define metadata dictionary
    metadata = {
        "legend_title": f"Brewing Process {brewing} - Tank {tank_id}",
        "x_label": "Time",
        "x_unit": "seconds",
        "y_label": "Inner Energy",
        "y_unit": "Joules",
        "experiment_date": "2025-02-17",
        "author": "HNoseery",
        "temperature_unit": "Celsius",
        "filter_sizes": str(filter_sizes)
    }

    from project.functions import store_plot_data
    # ================== Data Archiving ==================
    # Call store_plot_data() with final parameters
    output_dir = os.path.dirname(h5_path)
    os.makedirs(output_dir, exist_ok=True)

    store_plot_data(
        data=df_data,  # Processed data dictionary
        file_path=h5_path,  # From Aufgabe 5a
        group_path=group_path,  # From Aufgabe 5a
        metadata=metadata  # From Aufgabe 5a
    )

    print(f"\nData archived successfully to: {h5_path}")

    from project.functions import read_plot_data
    from project.functions import plot_data
    from project.functions import publish_plot

    df_plot, plot_labels = read_plot_data(
        file_path=h5_path,
        group_path=group_path
    )

    # 2. Generate plot
    fig = plot_data(df_plot, plot_labels)

    # 3. Publish results
    publish_plot(
        fig=fig,
        source_paths=[
            h5_path,  # Processed HDF5 file
            "/Users/HNoseery/Desktop/pa-ws2425/project/functions.py",  # Module
            "/Users/HNoseery/Desktop/pa-ws2425/main.py"  # Main script
        ],
        destination_path="./plotid"  # Output directory
    )

    print("\nPlot and files published to: ./plotid")



if __name__ == "__main__":
    main()

