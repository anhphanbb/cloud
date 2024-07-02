# import pandas as pd
# import os
# from netCDF4 import Dataset
# import cv2
# import numpy as np
# import random

# # Path to the CSV file with filenames
# csv_file_path = 'day_1_cloud_intervals.csv'

# # Read the CSV file
# data = pd.read_csv(csv_file_path)

# # Define output folders
# cloud_folder = 'images/cloud'
# clear_folder = 'images/clear'

# # Ensure output folders exist
# os.makedirs(cloud_folder, exist_ok=True)
# os.makedirs(clear_folder, exist_ok=True)

# # Function to check if a frame is within any of the cloud intervals
# def is_within_cloud_intervals(frame_index, cloud_intervals):
#     for interval in cloud_intervals:
#         if interval[0] <= frame_index <= interval[1]:
#             return True
#     return False

# # Function to read and save "Radiance" variable as images
# def save_radiance_as_images(nc_file_path, orbit_number, cloud_intervals, cloud_chance=0.15, clear_chance=0.06):
#     with Dataset(nc_file_path, 'r') as nc:
#         radiance = nc.variables['Radiance'][:]
#         for i in range(radiance.shape[0]):
#             if is_within_cloud_intervals(i, cloud_intervals):
#                 # 15% chance to save in 'cloud' folder
#                 if random.random() < cloud_chance:
#                     save_image(radiance[i], cloud_folder, orbit_number, i)
#             else:
#                 # 6% chance to save in 'clear' folder
#                 if random.random() < clear_chance:
#                     save_image(radiance[i], clear_folder, orbit_number, i)

# # Function to save a single image
# def save_image(data, folder, orbit_number, frame_index):
#     # Normalize the radiance data to fit in the range [0, 255]
#     norm_radiance = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
#     norm_radiance = norm_radiance.astype(np.uint8)
#     file_path = os.path.join(folder, f"orbit{orbit_number}_{frame_index}.png")
#     cv2.imwrite(file_path, norm_radiance)
#     print(f"Saved {file_path}")

# # Group the data by 'Oribt #' and collect intervals for each orbit
# grouped_data = data.groupby('Oribt #').apply(lambda x: list(zip(x['Start'], x['End']))).reset_index()
# grouped_data.columns = ['Oribt #', 'Intervals']

# # Read and save radiance images for all .nc files in the CSV
# for index, row in data.iterrows():
#     nc_file_path = row['File Name']
#     if pd.notna(nc_file_path):
#         orbit_number = row['Oribt #']
#         cloud_intervals = grouped_data[grouped_data['Oribt #'] == orbit_number]['Intervals'].values[0]
#         save_radiance_as_images(f'Day1/{nc_file_path}', orbit_number, cloud_intervals)

import pandas as pd
import os
from netCDF4 import Dataset
import cv2
import numpy as np
import random

# Path to the CSV file with filenames
csv_file_path = 'day_1_cloud_intervals.csv'

# Read the CSV file
data = pd.read_csv(csv_file_path)

# Define output folders
cloud_folder = 'images/cloud'
clear_folder = 'images/clear'

# Ensure output folders exist
os.makedirs(cloud_folder, exist_ok=True)
os.makedirs(clear_folder, exist_ok=True)

# Function to check if a frame is within any of the cloud intervals
def is_within_cloud_intervals(frame_index, cloud_intervals):
    for interval in cloud_intervals:
        if interval[0] <= frame_index <= interval[1]:
            return True
    return False

# Function to read and save "Radiance" variable as images
def save_radiance_as_images(nc_file_path, orbit_number, cloud_intervals, cloud_chance=0.15, clear_chance=0.06):
    with Dataset(nc_file_path, 'r') as nc:
        radiance = nc.variables['Radiance'][:]
        for i in range(radiance.shape[0]):
            if i < 5 or i >= radiance.shape[0] - 5:
                continue  # Skip if there are not enough frames before or after
            if is_within_cloud_intervals(i, cloud_intervals):
                # 15% chance to save in 'cloud' folder
                if random.random() < cloud_chance:
                    save_combined_image(radiance, i, cloud_folder, orbit_number)
            else:
                # 6% chance to save in 'clear' folder
                if random.random() < clear_chance:
                    save_combined_image(radiance, i, clear_folder, orbit_number)

# Function to save a single combined image
def save_combined_image(radiance, frame_index, folder, orbit_number):
    combined_image = np.hstack([
        normalize_image(radiance[frame_index - 5]),
        normalize_image(radiance[frame_index]),
        normalize_image(radiance[frame_index + 5])
    ])
    file_path = os.path.join(folder, f"orbit{orbit_number}_{frame_index}.png")
    cv2.imwrite(file_path, combined_image)
    print(f"Saved {file_path}")

# Function to normalize image data
def normalize_image(data):
    norm_radiance = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    return norm_radiance.astype(np.uint8)

# Group the data by 'Oribt #' and collect intervals for each orbit
grouped_data = data.groupby('Oribt #').apply(lambda x: list(zip(x['Start'], x['End']))).reset_index()
grouped_data.columns = ['Oribt #', 'Intervals']

# Read and save radiance images for all .nc files in the CSV
for index, row in data.iterrows():
    nc_file_path = row['File Name']
    if pd.notna(nc_file_path):
        orbit_number = row['Oribt #']
        cloud_intervals = grouped_data[grouped_data['Oribt #'] == orbit_number]['Intervals'].values[0]
        save_radiance_as_images(f'Day1/{nc_file_path}', orbit_number, cloud_intervals)
