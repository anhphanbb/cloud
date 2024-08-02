import os
import pandas as pd
import numpy as np
from netCDF4 import Dataset

# Define input and output folders
nc_input_folder = 'nc_files_to_predict'
predictions_folder = 'predictions'
nc_output_folder = 'nc_files_with_mlcloud'

# Ensure the output folder exists
os.makedirs(nc_output_folder, exist_ok=True)

# Function to extract orbit number from filename
def extract_orbit_number(filename):
    return int(filename.split('_')[3])

# Function to create a new filename by replacing l1r with l1c
def create_new_filename(filename):
    return filename.replace('l1r', 'l1c')

# Read .nc files and corresponding CSV files
for file_name in os.listdir(nc_input_folder):
    if file_name.endswith('.nc'):
        orbit_number = extract_orbit_number(file_name)
        nc_file_path = os.path.join(nc_input_folder, file_name)
        csv_file_path = os.path.join(predictions_folder, f'predictions_orbit_{orbit_number}.csv')
        
        if os.path.exists(csv_file_path):
            # Read and sort CSV file by frame number
            predictions_df = pd.read_csv(csv_file_path)
            predictions_df = predictions_df.sort_values(by='Frame #')
            
            # Add new variable 'MLCloud' using the running binary predictions
            mlcloud = predictions_df['Running Average Binary Prediction'].to_numpy()
            
            # Read the .nc file and create a new file with the 'MLCloud' variable
            new_nc_file_path = os.path.join(nc_output_folder, create_new_filename(file_name))
            with Dataset(nc_file_path, 'r') as src_nc, Dataset(new_nc_file_path, 'w') as dst_nc:
                # Copy dimensions
                for name, dimension in src_nc.dimensions.items():
                    dst_nc.createDimension(name, len(dimension) if not dimension.isunlimited() else None)
                
                # Copy variables
                for name, variable in src_nc.variables.items():
                    x = dst_nc.createVariable(name, variable.datatype, variable.dimensions)
                    dst_nc[name][:] = src_nc[name][:]
                
                # Add 'MLCloud' variable
                mlcloud_var = dst_nc.createVariable('MLCloud', 'i4', ('time',))
                mlcloud_var[:] = mlcloud
                print(f"Created new file with 'MLCloud' variable: {new_nc_file_path}")
        else:
            print(f"CSV file for orbit {orbit_number} not found in {predictions_folder}")

print("Processing completed.")
