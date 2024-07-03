import os
from netCDF4 import Dataset
import numpy as np

# Define input folder
input_folder = 'nc_files_to_predict'

# Function to read and print min and max "Radiance" values, ignoring NaNs
def print_radiance_min_max(nc_file_path):
    with Dataset(nc_file_path, 'r') as nc:
        radiance = nc.variables['Radiance'][:]
        min_value = np.nanmin(radiance)
        max_value = np.nanmax(radiance)
        print(f"File: {nc_file_path}, Min Radiance: {min_value}, Max Radiance: {max_value}")

# Function to parse orbit number from the file name (optional, if needed for additional processing)
def parse_orbit_number(file_name):
    parts = file_name.split('_')
    return int(parts[3])

# Read and print min and max radiance values for all .nc files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.nc'):
        nc_file_path = os.path.join(input_folder, file_name)
        print_radiance_min_max(nc_file_path)
