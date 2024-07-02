import pandas as pd
import netCDF4 as nc
from netCDF4 import Dataset

# Load the CSV file
file_path = 'day_1_cloud_intervals.csv'
csv_data = pd.read_csv(file_path)

# Get the first .nc file name from the CSV
first_nc_file = csv_data['File Name'].dropna().iloc[0]

# Path to the .nc file
nc_file_path = f'Day1/{first_nc_file}'

# Open the .nc file and display its variables
with nc.Dataset(nc_file_path) as dataset:
    print(f"Variables in the .nc file '{first_nc_file}':")
    for var_name in dataset.variables:
        print(var_name)
