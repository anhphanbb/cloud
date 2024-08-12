import os
import pandas as pd
import numpy as np
from netCDF4 import Dataset

# Define input and output folders
nc_input_folder = 'one_nc_file'
predictions_folder = 'predictions/3fsx3'
nc_output_folder = 'one_nc_file_with_mlcloud'

# Ensure the output folder exists
os.makedirs(nc_output_folder, exist_ok=True)

# Function to extract orbit number from filename
def extract_orbit_number(filename):
    return int(filename.split('_')[4])

# Function to create a new filename by replacing l1r with l1c
def create_new_filename(filename):
    return filename.replace('l1r', 'l1c')

# Function to extend MLCloud values with shifts
def extend_mlcloud_with_shifts(mlcloud):
    num_frames, num_boxes = mlcloud.shape
    extended_mlcloud = np.zeros((num_frames, 40))
    
    # Copy original boxes (0-8)
    extended_mlcloud[:, :9] = mlcloud[:, :9]
    
    # Define shifts for additional boxes
    shifts = {
        9: (0, 19), 10: (1, 19), 11: (2, 19),   # Boxes 0, 1, 2 shifted by +19 frames
        12: (0, 13), 13: (1, 13), 14: (2, 13),  # Boxes 0, 1, 2 shifted by +13 frames
        15: (0, 6), 16: (1, 6), 17: (2, 6),     # Boxes 0, 1, 2 shifted by +6 frames
        18: (0, -6), 19: (1, -6), 20: (2, -6),  # Boxes 0, 1, 2 shifted by -6 frames
        21: (3, 6), 22: (4, 6), 23: (5, 6),     # Boxes 3, 4, 5 shifted by +6 frames
        24: (3, -6), 25: (4, -6), 26: (5, -6),  # Boxes 3, 4, 5 shifted by -6 frames
        27: (6, 6), 28: (7, 6), 29: (8, 6),     # Boxes 6, 7, 8 shifted by +6 frames
        30: (6, -6), 31: (7, -6), 32: (8, -6),  # Boxes 6, 7, 8 shifted by -6 frames
        33: (6, -13), 34: (7, -13), 35: (8, -13), # Boxes 6, 7, 8 shifted by -13 frames
        36: (6, -19), 37: (7, -19), 38: (8, -19), # Boxes 6, 7, 8 shifted by -19 frames
        39: (7, -25) # Box 7 shifted by -25 frames
    }
    
    # Apply shifts
    for box_idx, (original_box, shift) in shifts.items():
        shifted_mlcloud = np.roll(mlcloud[:, original_box], shift)
        
        if shift > 0:
            # For positive shifts, fill the start with the value of the first valid frame
            shifted_mlcloud[:shift] = shifted_mlcloud[shift]
        elif shift < 0:
            # For negative shifts, fill the end with the value of the last valid frame
            shifted_mlcloud[shift:] = shifted_mlcloud[shift - 1]
        
        extended_mlcloud[:, box_idx] = shifted_mlcloud
    
    return extended_mlcloud

# Read .nc files and corresponding CSV files
for file_name in os.listdir(nc_input_folder):
    if file_name.endswith('.nc'):
        orbit_number = extract_orbit_number(file_name)
        nc_file_path = os.path.join(nc_input_folder, file_name)
        csv_file_path = os.path.join(predictions_folder, f'predictions_orbit_{orbit_number}.csv')
        
        if os.path.exists(csv_file_path):
            # Read and sort CSV file by frame number and box index
            predictions_df = pd.read_csv(csv_file_path)
            predictions_df = predictions_df.sort_values(by=['Frame #', 'Box Index'])
            
            # Pivot the DataFrame to have Frame # as the index and 9 columns for the boxes
            pivot_df = predictions_df.pivot(index='Frame #', columns='Box Index', values='Filtered Binary Prediction')
            
            # Convert the DataFrame to a NumPy array
            mlcloud = pivot_df.to_numpy()
            
            # Extend mlcloud with additional boxes and shifts
            extended_mlcloud = extend_mlcloud_with_shifts(mlcloud)
            
            # Extend mlcloud with edge frame values
            if len(extended_mlcloud) > 0:
                # Add 5 frames at the beginning with the prediction of the 6th frame
                start_frame_values = extended_mlcloud[5] if len(extended_mlcloud) > 5 else np.zeros(extended_mlcloud.shape[1])
                extended_mlcloud = np.vstack([np.tile(start_frame_values, (5, 1)), extended_mlcloud])
                
                # Add 5 frames at the end with the prediction of the second-to-last frame
                end_frame_values = extended_mlcloud[-6] if len(extended_mlcloud) > 5 else np.zeros(extended_mlcloud.shape[1])
                extended_mlcloud = np.vstack([extended_mlcloud, np.tile(end_frame_values, (5, 1))])
            
            # Read the .nc file and create a new file with the 'MLCloud' variable
            new_nc_file_path = os.path.join(nc_output_folder, create_new_filename(file_name))
            with Dataset(nc_file_path, 'r') as src_nc, Dataset(new_nc_file_path, 'w', format=src_nc.file_format) as dst_nc:
                # Copy global attributes
                dst_nc.setncatts({k: src_nc.getncattr(k) for k in src_nc.ncattrs()})
                
                # Copy dimensions
                for name, dimension in src_nc.dimensions.items():
                    dst_nc.createDimension(name, len(dimension) if not dimension.isunlimited() else None)
                
                # Create a new dimension for the 40 boxes
                dst_nc.createDimension('boxes', 40)
                
                # Copy variables
                for name, variable in src_nc.variables.items():
                    x = dst_nc.createVariable(name, variable.datatype, variable.dimensions, zlib=variable.filters().get('zlib', False))
                    dst_nc[name][:] = src_nc[name][:]
                    # Copy variable attributes
                    dst_nc[name].setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
                
                # Add 'MLCloud' variable
                mlcloud_var = dst_nc.createVariable('MLCloud', 'i4', ('time', 'boxes'), zlib=True)
                mlcloud_var[:] = extended_mlcloud
                print(f"Created new file with 'MLCloud' variable: {new_nc_file_path}")
        else:
            print(f"CSV file for orbit {orbit_number} not found in {predictions_folder}")

print("Processing completed.")
