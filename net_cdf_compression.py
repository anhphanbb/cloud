import os
import pandas as pd
import numpy as np
from netCDF4 import Dataset
import time
import matplotlib.pyplot as plt

# Define input and output folders
nc_input_folder = 'one_nc_file'
predictions_folder = 'predictions/3fsx3'
nc_output_folder = 'one_nc_file_with_mlcloud'

# Define the bounding boxes (all 40 boxes)
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

# Mapping for y_box_along_track and x_box_cross_track dimensions
box_mapping = [
    (3, 0), (3, 1), (3, 2), (6, 0), (6, 1), (6, 2), 
    (9, 0), (9, 1), (9, 2), (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), 
    (4, 0), (4, 1), (4, 2), (5, 0), (5, 1), (5, 2), 
    (7, 0), (7, 1), (7, 2), (8, 0), (8, 1), (8, 2), 
    (10, 0), (10, 1), (10, 2), (11, 0), (11, 1), (11, 2), 
    (12, 0), (12, 1), (12, 2), (13, 1), 
    (13, 0), (13, 2) # These two will have N/A values
]

# Ensure the output folder exists
os.makedirs(nc_output_folder, exist_ok=True)

# Function to extract orbit number from filename
def extract_orbit_number(filename):
    return int(filename.split('_')[4])

# Function to create a new filename by replacing l1r with l1c and adding compression level
def create_new_filename(filename, compression_level):
    return filename.replace('l1r', f'l1c_compression_{compression_level}')

# Function to extend MLCloud values with shifts (same as before)
def extend_mlcloud_with_shifts(mlcloud):
    num_frames, num_boxes = mlcloud.shape
    extended_mlcloud = np.zeros((num_frames, 3, 14))  # 14 columns (x_box_along_track) and 3 rows (y_box_cross_track)
    
    # Copy original boxes (0-8)
    for box_idx in range(9):
        x_idx, y_idx = box_mapping[box_idx]
        extended_mlcloud[:, y_idx, x_idx] = mlcloud[:, box_idx]
    
    # Define shifts for additional boxes
    shifts = {
        9: (0, 19), 10: (1, 19), 11: (2, 19),
        12: (0, 13), 13: (1, 13), 14: (2, 13),
        15: (0, 6), 16: (1, 6), 17: (2, 6),
        18: (0, -6), 19: (1, -6), 20: (2, -6),
        21: (3, 6), 22: (4, 6), 23: (5, 6),
        24: (3, -6), 25: (4, -6), 26: (5, -6),
        27: (6, 6), 28: (7, 6), 29: (8, 6),
        30: (6, -6), 31: (7, -6), 32: (8, -6),
        33: (6, -13), 34: (7, -13), 35: (8, -13),
        36: (6, -19), 37: (7, -19), 38: (8, -19),
        39: (7, -25)
    }
    
    # Apply shifts
    for box_idx, (original_box, shift) in shifts.items():
        shifted_mlcloud = np.roll(mlcloud[:, original_box], shift)
        
        if shift > 0:
            shifted_mlcloud[:shift] = shifted_mlcloud[shift]
        elif shift < 0:
            shifted_mlcloud[shift:] = shifted_mlcloud[shift - 1]
        
        x_idx, y_idx = box_mapping[box_idx]
        extended_mlcloud[:, y_idx, x_idx] = shifted_mlcloud
    
    return extended_mlcloud

# Initialize lists to track compression times and file sizes
compression_times = []
file_sizes = []
compression_levels = list(range(10))

# Read .nc files and corresponding CSV files
for file_name in os.listdir(nc_input_folder):
    if file_name.endswith('.nc'):
        orbit_number = extract_orbit_number(file_name)
        nc_file_path = os.path.join(nc_input_folder, file_name)
        csv_file_path = os.path.join(predictions_folder, f'predictions_orbit_{orbit_number}.csv')
        
        if os.path.exists(csv_file_path):
            predictions_df = pd.read_csv(csv_file_path)
            predictions_df = predictions_df.sort_values(by=['Frame #', 'Box Index'])
            pivot_df = predictions_df.pivot(index='Frame #', columns='Box Index', values='Running Average Probability')
            mlcloud = pivot_df.to_numpy()
            mlcloud = np.round(mlcloud * 100).astype(np.int32)
            extended_mlcloud = extend_mlcloud_with_shifts(mlcloud)
            
            if len(extended_mlcloud) > 0:
                start_frame_values = extended_mlcloud[5] if len(extended_mlcloud) > 5 else np.zeros(extended_mlcloud.shape[1:])
                extended_mlcloud = np.vstack([np.tile(start_frame_values, (5, 1, 1)), extended_mlcloud])
                end_frame_values = extended_mlcloud[-6] if len(extended_mlcloud) > 5 else np.zeros(extended_mlcloud.shape[1:])
                extended_mlcloud = np.vstack([extended_mlcloud, np.tile(end_frame_values, (5, 1, 1))])
            
            # Loop through compression levels (0-9)
            for compression_level in compression_levels:
                start_time = time.time()
                new_nc_file_path = os.path.join(nc_output_folder, create_new_filename(file_name, compression_level))
                with Dataset(nc_file_path, 'r') as src_nc, Dataset(new_nc_file_path, 'w', format=src_nc.file_format) as dst_nc:
                    dst_nc.setncatts({k: src_nc.getncattr(k) for k in src_nc.ncattrs()})
                    for name, dimension in src_nc.dimensions.items():
                        dst_nc.createDimension(name, len(dimension) if not dimension.isunlimited() else None)
                    dst_nc.createDimension('x_box_along_track', 14)
                    dst_nc.createDimension('y_box_cross_track', 3)
                    for name, variable in src_nc.variables.items():
                        x = dst_nc.createVariable(name, variable.datatype, variable.dimensions, zlib=compression_level > 0, complevel=compression_level)
                        dst_nc[name][:] = src_nc[name][:]
                        dst_nc[name].setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
                    mlcloud_var = dst_nc.createVariable('MLCloud', 'i4', ('time', 'y_box_cross_track', 'x_box_along_track'), zlib=True, complevel=compression_level)
                    mlcloud_var[:] = extended_mlcloud
                
                end_time = time.time()
                compression_time = end_time - start_time
                file_size = os.path.getsize(new_nc_file_path)
                
                # Store compression time and file size for each level
                compression_times.append(compression_time)
                file_sizes.append(file_size / 1024)  # Convert bytes to kilobytes

                print(f"Compression Level: {compression_level} - Time: {compression_time:.2f}s, File Size: {file_size / 1024:.2f}KB")

# Plot compression times and file sizes
plt.figure(figsize=(10, 6))
plt.plot(compression_levels, compression_times, label='Compression Time (s)', color='b', marker='o')
plt.plot(compression_levels, file_sizes, label='File Size (KB)', color='g', marker='x')
plt.title('Compression Time and File Size vs. Compression Level')
plt.xlabel('Compression Level')
plt.ylabel('Time (s) / Size (KB)')
plt.legend()
plt.grid(True)
plt.show()
