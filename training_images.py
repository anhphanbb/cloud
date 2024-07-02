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
cloud_flow_folder = 'images/cloud_flow'
clear_flow_folder = 'images/clear_flow'

# Ensure output folders exist
os.makedirs(cloud_folder, exist_ok=True)
os.makedirs(clear_folder, exist_ok=True)
os.makedirs(cloud_flow_folder, exist_ok=True)
os.makedirs(clear_flow_folder, exist_ok=True)

# Function to check if a frame is within any of the cloud intervals
def is_within_cloud_intervals(frame_index, cloud_intervals):
    for interval in cloud_intervals:
        if interval[0] <= frame_index <= interval[1]:
            return True
    return False

# Function to calculate optical flow between two frames
def calculate_optical_flow(prev_frame, next_frame):
    prev_gray = cv2.normalize(prev_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    next_gray = cv2.normalize(next_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    optical_flow_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return optical_flow_image

# Function to read and save "Radiance" variable as images and their optical flow
def save_radiance_and_optical_flow_as_images(nc_file_path, orbit_number, cloud_intervals, cloud_chance=0.15, clear_chance=0.06):
    with Dataset(nc_file_path, 'r') as nc:
        radiance = nc.variables['Radiance'][:]
        prev_frame = None
        for i in range(radiance.shape[0]):
            current_frame = radiance[i]
            if is_within_cloud_intervals(i, cloud_intervals):
                if random.random() < cloud_chance:
                    save_image(current_frame, cloud_folder, orbit_number, i)
                    if prev_frame is not None:
                        optical_flow_image = calculate_optical_flow(prev_frame, current_frame)
                        save_optical_flow_image(optical_flow_image, cloud_flow_folder, orbit_number, i)
            else:
                if random.random() < clear_chance:
                    save_image(current_frame, clear_folder, orbit_number, i)
                    if prev_frame is not None:
                        optical_flow_image = calculate_optical_flow(prev_frame, current_frame)
                        save_optical_flow_image(optical_flow_image, clear_flow_folder, orbit_number, i)
            prev_frame = current_frame

# Function to save a single image
def save_image(data, folder, orbit_number, frame_index):
    norm_radiance = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    norm_radiance = norm_radiance.astype(np.uint8)
    file_path = os.path.join(folder, f"orbit{orbit_number}_{frame_index}.png")
    cv2.imwrite(file_path, norm_radiance)
    print(f"Saved {file_path}")

# Function to save a single optical flow image
def save_optical_flow_image(optical_flow_image, folder, orbit_number, frame_index):
    file_path = os.path.join(folder, f"orbit{orbit_number}_{frame_index}.png")
    cv2.imwrite(file_path, optical_flow_image)
    print(f"Saved {file_path}")

# Group the data by 'Oribt #' and collect intervals for each orbit
grouped_data = data.groupby('Oribt #').apply(lambda x: list(zip(x['Start'], x['End']))).reset_index()
grouped_data.columns = ['Oribt #', 'Intervals']

# Read and save radiance images and their optical flow for all .nc files in the CSV
for index, row in data.iterrows():
    nc_file_path = row['File Name']
    if pd.notna(nc_file_path):
        orbit_number = row['Oribt #']
        cloud_intervals = grouped_data[grouped_data['Oribt #'] == orbit_number]['Intervals'].values[0]
        save_radiance_and_optical_flow_as_images(f'Day1/{nc_file_path}', orbit_number, cloud_intervals)
