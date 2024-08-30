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

boxes = [
    (60, 80, 0, 100),     # Box 0
    (60, 80, 100, 200),   # Box 1
    (60, 80, 200, 300),   # Box 2
    (120, 140, 0, 100),   # Box 3
    (120, 140, 100, 200), # Box 4
    (120, 140, 200, 300), # Box 5
    (180, 200, 0, 100),   # Box 6
    (180, 200, 100, 200), # Box 7
    (180, 200, 200, 300), # Box 8
    (0, 20, 0, 100),      # Box 9
    (0, 20, 100, 200),    # Box 10
    (0, 20, 200, 300),    # Box 11
    (20, 40, 0, 100),     # Box 12
    (20, 40, 100, 200),   # Box 13
    (20, 40, 200, 300),   # Box 14
    (40, 60, 0, 100),     # Box 15
    (40, 60, 100, 200),   # Box 16
    (40, 60, 200, 300),   # Box 17
    (80, 100, 0, 100),    # Box 18
    (80, 100, 100, 200),  # Box 19
    (80, 100, 200, 300),  # Box 20
    (100, 120, 0, 100),   # Box 21
    (100, 120, 100, 200), # Box 22
    (100, 120, 200, 300), # Box 23
    (140, 160, 0, 100),   # Box 24
    (140, 160, 100, 200), # Box 25
    (140, 160, 200, 300), # Box 26
    (160, 180, 0, 100),   # Box 27
    (160, 180, 100, 200), # Box 28
    (160, 180, 200, 300), # Box 29
    (200, 220, 0, 100),   # Box 30
    (200, 220, 100, 200), # Box 31
    (200, 220, 200, 300), # Box 32
    (220, 240, 0, 100),   # Box 33
    (220, 240, 100, 200), # Box 34
    (220, 240, 200, 300), # Box 35
    (240, 260, 0, 100),   # Box 36
    (240, 260, 100, 200), # Box 37
    (240, 260, 200, 300), # Box 38
    (260, 280, 100, 200), # Box 39
]

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
            pivot_df = predictions_df.pivot(index='Frame #', columns='Box Index', values='Running Average Probability')
            
            # Convert the DataFrame to a NumPy array
            mlcloud = pivot_df.to_numpy()
            
            # Scale the probabilities from 0-1 to 0-100
            mlcloud = np.round(mlcloud * 100).astype(np.int32)
            
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
                
                # Extract latitude and longitude data
                lat_data = src_nc.variables['Latitude'][:]
                lon_data = src_nc.variables['Longitude'][:]
                
                # Calculate the exact center coordinates for each box
                box_centers = [(int((x_start + x_end) / 2), int((y_start + y_end) / 2)) for x_start, x_end, y_start, y_end in boxes]
                
                # Extract center latitudes and longitudes for each box and time frame
                box_center_latitudes = np.zeros((lat_data.shape[0], 40))
                box_center_longitudes = np.zeros((lon_data.shape[0], 40))

                # Assign latitude and longitude values of the center pixel to each box
                for i, (x_center, y_center) in enumerate(box_centers):
                    box_center_latitudes[:, i] = lat_data[:, y_center, x_center]
                    box_center_longitudes[:, i] = lon_data[:, y_center, x_center]
                
                # Add 'MLCloud' variable
                mlcloud_var = dst_nc.createVariable('MLCloud', 'i4', ('time', 'boxes'), zlib=True)
                mlcloud_var[:] = extended_mlcloud

                # Create variables for box center coordinates
                lat_var = dst_nc.createVariable('Box_Center_Latitude', 'f4', ('time', 'boxes'), zlib=True)
                lon_var = dst_nc.createVariable('Box_Center_Longitude', 'f4', ('time', 'boxes'), zlib=True)

                # Assign the latitude and longitude of box centers to the variables
                lat_var[:] = box_center_latitudes
                lon_var[:] = box_center_longitudes
                
                print(f"Created new file with 'MLCloud' variable and box center coordinates: {new_nc_file_path}")
        else:
            print(f"CSV file for orbit {orbit_number} not found in {predictions_folder}")

print("Processing completed.")
