import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# Define the output folder for the new .nc files
nc_output_folder = 'nc_files_with_mlcloud'

# Function to extract orbit number from filename
def extract_orbit_number(filename):
    return int(filename.split('_')[3])

# Read the new l1c .nc files and plot the MLCloud values
for file_name in os.listdir(nc_output_folder):
    if file_name.endswith('.nc'):
        orbit_number = extract_orbit_number(file_name)
        nc_file_path = os.path.join(nc_output_folder, file_name)
        
        with Dataset(nc_file_path, 'r') as nc:
            # Check if 'MLCloud' variable exists
            if 'MLCloud' in nc.variables:
                mlcloud = nc.variables['MLCloud'][:]
                
                # Plot MLCloud values
                plt.figure(figsize=(12, 3))
                plt.plot(mlcloud, label='MLCloud')
                plt.title(f'MLCloud Values for Orbit {orbit_number}')
                plt.xlabel('Frame #')
                plt.ylabel('MLCloud Value')
                plt.legend()
                plt.show()
            else:
                print(f"'MLCloud' variable not found in {nc_file_path}")

print("Plotting completed.")
