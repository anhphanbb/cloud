# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:56:16 2024

@author: Anh

Update Aug 30 2024: Reads the orbit number from the CSV file and searches for the corresponding .nc file in the directory structure
"""

import pandas as pd
import os
from netCDF4 import Dataset
import cv2
import numpy as np
import random
import re

# Path to the CSV file with filenames
csv_file_path = 'csv/newest/cloud_intervals_3fs_aug_30.csv'
parent_directory = 'l1r_11_updated_07032024'

# Read the CSV file
data = pd.read_csv(csv_file_path)

# Ensure orbit numbers are integers and filter out NaN values
data = data.dropna(subset=['Orbit #'])  # Remove rows where 'Orbit #' is NaN
data['Orbit #'] = data['Orbit #'].astype(int)  # Convert orbit numbers to integers

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

# Function to search for the correct .nc file based on the orbit number
def find_nc_file(parent_directory, orbit_number):
    orbit_str = str(int(orbit_number)).zfill(5)  # Ensure orbit number is an integer and properly padded
    pattern = re.compile(r'awe_l1r_(.*)_' + orbit_str + r'_(.*)\.nc')
    for root, dirs, files in os.walk(parent_directory):
        for file in files:
            if pattern.match(file):
                return os.path.join(root, file)
    raise FileNotFoundError(f"No file found for orbit number {orbit_str}")

# Function to read and save "Radiance" variable as images
def save_radiance_as_images(nc_file_path, orbit_number, cloud_intervals_list, cloud_chance=0.25, clear_chance=0.02):
    with Dataset(nc_file_path, 'r') as nc:
        radiance = nc.variables['Radiance'][:]
        num_frames = radiance.shape[0]
        
        for box_idx, intervals in enumerate(cloud_intervals_list):
            # Print the intervals once per box
            print(f"Orbit {orbit_number}, Box {box_idx}: Intervals = {intervals}")
            
            for i in range(5, num_frames - 5):
                for interval in intervals:
                    if is_within_cloud_intervals(i, interval):
                        if random.random() < cloud_chance:
                            save_image(radiance, cloud_folder, orbit_number, i, box_idx)
                            if i >= 18:
                                save_image(radiance, cloud_folder, orbit_number, i - 13, box_idx + 3)
                            if i < num_frames - 25:
                                save_image(radiance, cloud_folder, orbit_number, i + 19, box_idx + 6)
                    else:
                        if random.random() < clear_chance:
                            save_image(radiance, clear_folder, orbit_number, i, box_idx)
                            if i >= 18:
                                save_image(radiance, clear_folder, orbit_number, i - 13, box_idx + 3)
                            if i < num_frames - 25:
                                save_image(radiance, clear_folder, orbit_number, i + 19, box_idx + 6)

# Function to save three frames as a single image
def save_image(data, folder, orbit_number, frame_index, box_idx):
    min_radiance = 0
    max_radiance = 24

    norm_radiance = np.clip((data[frame_index] - min_radiance) / (max_radiance - min_radiance) * 255, 0, 255)
    norm_radiance = np.nan_to_num(norm_radiance, nan=0, posinf=255, neginf=0).astype(np.uint8)

    prev_frame_norm = None
    if frame_index >= 5:
        prev_frame = data[frame_index - 5]
        prev_frame_norm = np.clip((prev_frame - min_radiance) / (max_radiance - min_radiance) * 255, 0, 255)
        prev_frame_norm = np.nan_to_num(prev_frame_norm, nan=0, posinf=255, neginf=0).astype(np.uint8)

    next_frame_norm = None
    if frame_index < data.shape[0] - 5:
        next_frame = data[frame_index + 5]
        next_frame_norm = np.clip((next_frame - min_radiance) / (max_radiance - min_radiance) * 255, 0, 255)
        next_frame_norm = np.nan_to_num(next_frame_norm, nan=0, posinf=255, neginf=0).astype(np.uint8)

    x_start, x_end, y_start, y_end = boxes[box_idx]

    three_layer_image = np.zeros((data.shape[1], data.shape[2], 3), dtype=np.uint8)
    if prev_frame_norm is not None:
        three_layer_image[..., 0] = prev_frame_norm
    three_layer_image[..., 1] = norm_radiance
    if next_frame_norm is not None:
        three_layer_image[..., 2] = next_frame_norm

    cropped_image = three_layer_image[y_start:y_end, x_start:x_end]

    file_path = os.path.join(folder, f"orbit{orbit_number}_box{box_idx}_{frame_index}.png")
    cv2.imwrite(file_path, cropped_image)

# Function to create interval lists from the data
def create_intervals_list(row):
    intervals_list = [[] for _ in range(3)]
    
    if pd.notna(row['Start1']) and pd.notna(row['End1']):
        intervals_list[0].append((int(row['Start1']), int(row['End1'])))
    if pd.notna(row['Start2']) and pd.notna(row['End2']):
        intervals_list[1].append((int(row['Start2']), int(row['End2'])))
    if pd.notna(row['Start3']) and pd.notna(row['End3']):
        intervals_list[2].append((int(row['Start3']), int(row['End3'])))
    
    return intervals_list

# Initialize a dictionary to store intervals for each orbit
orbit_intervals = {}

# Read and save radiance images for all .nc files in the CSV
for index, row in data.iterrows():
    orbit_number = row['Orbit #']
    if pd.notna(orbit_number):
        orbit_number = int(orbit_number)  # Ensure orbit number is an integer
        
        # Create the intervals list for this row
        intervals = create_intervals_list(row)
        
        if orbit_number not in orbit_intervals:
            orbit_intervals[orbit_number] = [[] for _ in range(3)]
        
        # Append the intervals from this row to the orbit's interval lists
        for i in range(3):
            orbit_intervals[orbit_number][i].extend(intervals[i])

# Now process the intervals for each orbit and save images
for orbit_number, cloud_intervals_list in orbit_intervals.items():
    try:
        nc_file_path = find_nc_file(parent_directory, orbit_number)
    except FileNotFoundError as e:
        print(e)
        continue
    
    save_radiance_as_images(nc_file_path, orbit_number, cloud_intervals_list)
