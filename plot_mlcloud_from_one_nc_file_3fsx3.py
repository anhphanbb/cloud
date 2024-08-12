import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# Define the path to the NetCDF file
nc_file_path = 'one_nc_file_with_mlcloud/awe_l1c_q20_2023330T1332_00090_v01.nc' 

# Load the NetCDF file
with Dataset(nc_file_path, 'r') as nc_file:
    # Read the MLCloud variable
    mlcloud = nc_file.variables['MLCloud'][:]
    
    # Check the dimensions of the MLCloud variable
    print(f"MLCloud shape: {mlcloud.shape}")  # Should be (time, boxes)
    
    # Number of time steps and boxes
    num_frames, num_boxes = mlcloud.shape
    
    # Create a figure with 9 subplots
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten the 2D array of axes to make indexing easier
    
    for box_idx in range(num_boxes):
        ax = axes[box_idx]
        ax.plot(range(num_frames), mlcloud[:, box_idx], label=f'Box {box_idx}')
        ax.set_title(f'Box {box_idx}')
        ax.set_xlabel('Frame #')
        ax.set_ylabel('MLCloud Value')
        ax.grid(True)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
