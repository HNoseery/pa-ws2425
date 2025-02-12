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



if __name__ == "__main__":
    main()

