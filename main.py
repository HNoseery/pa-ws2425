import datetime
import os

import numpy as np

import project.functions as fn
from project.functions import read_metadata


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

    df_data = {}




if __name__ == "__main__":
    main()

