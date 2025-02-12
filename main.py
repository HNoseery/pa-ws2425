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




if __name__ == "__main__":
    main()

