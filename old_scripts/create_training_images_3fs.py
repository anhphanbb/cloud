# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:56:16 2024

@author: Anh
"""

import pandas as pd
import os
from netCDF4 import Dataset
import cv2
import numpy as np
import random

# Path to the CSV file with filenames
csv_file_path = 'csv/cloud_intervals_3fs_aug_7.csv'

# Read the CSV file
data = pd.read_csv(csv_file_path)

# Print the column names to debug KeyError
print("Column names in CSV:", data.columns)

# Define output folders
cloud_folder = 'images/cloud'
clear_folder = 'images/clear'

# Ensure output folders exist
os.makedirs(cloud_folder, exist_ok=True)
os.makedirs(clear_folder, exist_ok=True)

# Bounding box coordinates for three smaller boxes
boxes = [
    (140, 160, 0, 100),     # Bottom
    (140, 160, 100, 200),   # Middle
    (140, 160, 200, 300)    # Top
]

# Function to check if a frame is within any of the cloud intervals
def is_within_cloud_intervals(frame_index, cloud_intervals):
    for interval in cloud_intervals:
        if interval[0] <= frame_index <= interval[1]:
            return True
    return False

# Function to read and save "Radiance" variable as images
def save_radiance_as_images(nc_file_path, orbit_number, cloud_intervals_list, cloud_chance=0.25, clear_chance=0.02):
    with Dataset(nc_file_path, 'r') as nc:
        radiance = nc.variables['Radiance'][:]
        for i in range(5, radiance.shape[0] - 5):
            for box_idx, cloud_intervals in enumerate(cloud_intervals_list):
                if is_within_cloud_intervals(i, cloud_intervals):
                    # 25% chance to save in 'cloud' folder
                    if random.random() < cloud_chance:
                        save_image(radiance, cloud_folder, orbit_number, i, box_idx)
                else:
                    # 6% chance to save in 'clear' folder
                    if random.random() < clear_chance:
                        save_image(radiance, clear_folder, orbit_number, i, box_idx)

# Function to save three frames as a single image
def save_image(data, folder, orbit_number, frame_index, box_idx):
    # Normalize the radiance data to fit in the range [0, 255] using fixed min and max values
    min_radiance = 0
    max_radiance = 24

    # Normalize current frame radiance data
    norm_radiance = np.clip((data[frame_index] - min_radiance) / (max_radiance - min_radiance) * 255, 0, 255).astype(np.uint8)
    
    # Normalize previous frame radiance data
    prev_frame_norm = None
    if frame_index >= 5:
        prev_frame = data[frame_index - 5]
        prev_frame_norm = np.clip((prev_frame - min_radiance) / (max_radiance - min_radiance) * 255, 0, 255).astype(np.uint8)
    
    # Normalize next frame radiance data
    next_frame_norm = None
    if frame_index < data.shape[0] - 5:
        next_frame = data[frame_index + 5]
        next_frame_norm = np.clip((next_frame - min_radiance) / (max_radiance - min_radiance) * 255, 0, 255).astype(np.uint8)
    
    # Get the bounding box coordinates
    x_start, x_end, y_start, y_end = boxes[box_idx]

    # Create a three-layer image
    three_layer_image = np.zeros((data.shape[1], data.shape[2], 3), dtype=np.uint8)
    if prev_frame_norm is not None:
        three_layer_image[..., 0] = prev_frame_norm
    three_layer_image[..., 1] = norm_radiance
    if next_frame_norm is not None:
        three_layer_image[..., 2] = next_frame_norm

    # Crop the image to the bounding box
    cropped_image = three_layer_image[y_start:y_end, x_start:x_end]

    file_path = os.path.join(folder, f"orbit{orbit_number}_box{box_idx}_{frame_index}.png")
    cv2.imwrite(file_path, cropped_image)
    print(f"Saved {file_path}")

# Group the data by the correct column name for 'Orbit #' and collect intervals for each orbit
correct_orbit_column_name = 'Orbit #'  # Use the correct column name based on the debug print output

# Function to create interval lists from the data
def create_intervals_list(row):
    intervals_list = []
    for i in range(1, 4):
        start_col = f'Start{i}'
        end_col = f'End{i}'
        if pd.notna(row[start_col]) and pd.notna(row[end_col]):
            intervals_list.append((int(row[start_col]), int(row[end_col])))
    return intervals_list

# Read and save radiance images for all .nc files in the CSV
for index, row in data.iterrows():
    nc_file_path = row['File Name']
    if pd.notna(nc_file_path):
        orbit_number = row[correct_orbit_column_name]
        cloud_intervals_list = []
        for i in range(1, 4):
            cloud_intervals_list.append(create_intervals_list(row))
        save_radiance_as_images(f'l1r_11_updated_07032024/{nc_file_path}', orbit_number, cloud_intervals_list)
