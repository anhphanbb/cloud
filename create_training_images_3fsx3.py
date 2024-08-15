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
csv_file_path = 'csv/newest/cloud_intervals_3fs_aug_13.csv'

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

# Bounding box coordinates for nine smaller boxes
boxes = [
    (140, 160, 0, 100),     # Bottom Row, Center Column
    (140, 160, 100, 200),   # Middle Row, Center Column
    (140, 160, 200, 300),   # Top Row, Center Column
    (180, 200, 0, 100),     # Bottom Row, Center +2 Column
    (180, 200, 100, 200),   # Middle Row, Center +2 Column
    (180, 200, 200, 300),   # Top Row, Center +2 Column
    (80, 100, 0, 100),      # Bottom Row, Center -3 Column
    (80, 100, 100, 200),    # Middle Row, Center -3 Column
    (80, 100, 200, 300)     # Top Row, Center -3 Column
]

# Function to check if a frame is within any of the cloud intervals
def is_within_cloud_intervals(frame_index, interval):
    return interval[0] <= frame_index <= interval[1]

# Function to read and save "Radiance" variable as images
def save_radiance_as_images(nc_file_path, orbit_number, cloud_intervals_list, cloud_chance=0.25, clear_chance=0.02):
    with Dataset(nc_file_path, 'r') as nc:
        radiance = nc.variables['Radiance'][:]
        num_frames = radiance.shape[0]
        
        for box_idx, intervals in enumerate(cloud_intervals_list):
            # Print the intervals once per box
            print(f"Orbit {orbit_number}, Box {box_idx}: Intervals = {intervals}")
            
            # for i in range(5, num_frames - 5):
            #     for interval in intervals:
            #         if is_within_cloud_intervals(i, interval):
            #             # 25% chance to save in 'cloud' folder
            #             if random.random() < cloud_chance:
            #                 save_image(radiance, cloud_folder, orbit_number, i, box_idx)
            #             # For i >= 18 (13 frames for clouds to move one box to another + 5 frames before for 3 layers), save image of radiance 6 frames before with box_idx + 3
            #             if i >= 18:
            #                 if random.random() < cloud_chance:
            #                     save_image(radiance, cloud_folder, orbit_number, i - 13, box_idx + 3)
            #             if i < num_frames - 25: #(10 frames for clouds to move one box to another + 5 frames before for 3 layers)
            #                 if random.random() < cloud_chance:
            #                     save_image(radiance, cloud_folder, orbit_number, i + 19, box_idx + 3)
            #         else:
            #             # 2% chance to save in 'clear' folder
            #             if random.random() < clear_chance:
            #                 save_image(radiance, clear_folder, orbit_number, i, box_idx)
                            
            #             if i >= 18:
            #                 if random.random() < clear_chance:
            #                     save_image(radiance, clear_folder, orbit_number, i - 13, box_idx + 3)
            #             if i < num_frames - 25:
            #                 if random.random() < clear_chance:
            #                     save_image(radiance, clear_folder, orbit_number, i + 19, box_idx + 3)
            
            for i in range(5, num_frames - 5):
                for interval in intervals:
                    if is_within_cloud_intervals(i, interval):
                        # 25% chance to save in 'cloud' folder
                        if random.random() < cloud_chance:
                            save_image(radiance, cloud_folder, orbit_number, i, box_idx)
                        # For i >= 18 (13 frames for clouds to move one box to another + 5 frames before for 3 layers), save image of radiance 6 frames before with box_idx + 3
                            if i >= 18:
                                save_image(radiance, cloud_folder, orbit_number, i - 13, box_idx + 3)   # right box: idx+3
                            if i < num_frames - 25: #(19 frames for clouds to move one box to another + 5 frames before for 3 layers)
                                save_image(radiance, cloud_folder, orbit_number, i + 19, box_idx + 6)   # left box: idx+6
                    else:
                        # 2% chance to save in 'clear' folder
                        if random.random() < clear_chance:
                            save_image(radiance, clear_folder, orbit_number, i, box_idx)
                            
                            if i >= 18:
                                save_image(radiance, clear_folder, orbit_number, i - 13, box_idx + 3)
                            if i < num_frames - 25:
                                save_image(radiance, clear_folder, orbit_number, i + 19, box_idx + 6)

# Function to save three frames as a single image
def save_image(data, folder, orbit_number, frame_index, box_idx):
    # Normalize the radiance data to fit in the range [0, 255] using fixed min and max values
    min_radiance = 0
    max_radiance = 24

    # Normalize current frame radiance data
    norm_radiance = np.clip((data[frame_index] - min_radiance) / (max_radiance - min_radiance) * 255, 0, 255)
    norm_radiance = np.nan_to_num(norm_radiance, nan=0, posinf=255, neginf=0).astype(np.uint8)

    # Normalize previous frame radiance data
    prev_frame_norm = None
    if frame_index >= 5:
        prev_frame = data[frame_index - 5]
        prev_frame_norm = np.clip((prev_frame - min_radiance) / (max_radiance - min_radiance) * 255, 0, 255)
        prev_frame_norm = np.nan_to_num(prev_frame_norm, nan=0, posinf=255, neginf=0).astype(np.uint8)

    # Normalize next frame radiance data
    next_frame_norm = None
    if frame_index < data.shape[0] - 5:
        next_frame = data[frame_index + 5]
        next_frame_norm = np.clip((next_frame - min_radiance) / (max_radiance - min_radiance) * 255, 0, 255)
        next_frame_norm = np.nan_to_num(next_frame_norm, nan=0, posinf=255, neginf=0).astype(np.uint8)

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
    #print(f"Saved {file_path}")


# Group the data by the correct column name for 'Orbit #' and collect intervals for each orbit
correct_orbit_column_name = 'Orbit #'  # Use the correct column name based on the debug print output

# Function to create interval lists from the data
def create_intervals_list(row):
    intervals_list = [[] for _ in range(3)]  # Create an empty list for each box in correct order
    
    # For each box, assign the corresponding Start/End pair if they exist
    if pd.notna(row['Start1']) and pd.notna(row['End1']):
        intervals_list[0].append((int(row['Start1']), int(row['End1'])))  # Box 0 -> Start1/End1
    if pd.notna(row['Start2']) and pd.notna(row['End2']):
        intervals_list[1].append((int(row['Start2']), int(row['End2'])))  # Box 1 -> Start2/End2
    if pd.notna(row['Start3']) and pd.notna(row['End3']):
        intervals_list[2].append((int(row['Start3']), int(row['End3'])))  # Box 2 -> Start3/End3
    
    return intervals_list

# Initialize a dictionary to store intervals for each orbit
orbit_intervals = {}

# Read and save radiance images for all .nc files in the CSV
for index, row in data.iterrows():
    nc_file_path = row['File Name']
    if pd.notna(nc_file_path):
        orbit_number = row[correct_orbit_column_name]
        
        # Create the intervals list for this row
        intervals = create_intervals_list(row)
        
        # If this orbit is not yet in the dictionary, add it
        if orbit_number not in orbit_intervals:
            orbit_intervals[orbit_number] = [[] for _ in range(3)]  # Initialize lists for each box
        
        # Append the intervals from this row to the orbit's interval lists
        for i in range(3):
            orbit_intervals[orbit_number][i].extend(intervals[i])

# Now process the intervals for each orbit and save images
for orbit_number, cloud_intervals_list in orbit_intervals.items():
    # Get the corresponding .nc file path for the orbit number
    nc_file_path = data[data[correct_orbit_column_name] == orbit_number]['File Name'].values[0]
    
    # Save the radiance images based on the collected intervals for this orbit
    save_radiance_as_images(f'l1r_11_updated_07032024/{nc_file_path}', orbit_number, cloud_intervals_list)
