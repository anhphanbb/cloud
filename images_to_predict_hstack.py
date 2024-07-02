# import pandas as pd
# import os
# from netCDF4 import Dataset
# import cv2
# import numpy as np

# # Define input and output folders
# input_folder = 'nc_files_to_predict'
# output_folder = 'images_to_predict'

# # Ensure the output folder exists
# os.makedirs(output_folder, exist_ok=True)

# # Function to save a single image
# def save_image(data, folder, orbit_number, frame_index):
#     # Normalize the radiance data to fit in the range [0, 255]
#     norm_radiance = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
#     norm_radiance = norm_radiance.astype(np.uint8)
#     file_path = os.path.join(folder, f"orbit{orbit_number}_{frame_index}.png")
#     cv2.imwrite(file_path, norm_radiance)
#     print(f"Saved {file_path}")

# # Function to read and save "Radiance" variable as images
# def save_radiance_as_images(nc_file_path, orbit_number):
#     with Dataset(nc_file_path, 'r') as nc:
#         radiance = nc.variables['Radiance'][:]
#         for i in range(radiance.shape[0]):
#             save_image(radiance[i], output_folder, orbit_number, i)

# # Function to parse orbit number from the file name
# def parse_orbit_number(file_name):
#     parts = file_name.split('_')
#     return int(parts[3])

# # Read and save radiance images for all .nc files in the input folder
# for file_name in os.listdir(input_folder):
#     if file_name.endswith('.nc'):
#         nc_file_path = os.path.join(input_folder, file_name)
#         orbit_number = parse_orbit_number(file_name)
#         save_radiance_as_images(nc_file_path, orbit_number)

import pandas as pd
import os
from netCDF4 import Dataset
import cv2
import numpy as np

# Define input and output folders
input_folder = 'nc_files_to_predict'
output_folder = 'images_to_predict'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to save a single combined image
def save_combined_image(radiance, frame_index, folder, orbit_number):
    if frame_index < 5 or frame_index >= len(radiance) - 5:
        return  # Skip if there are not enough frames before or after
    
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

# Function to read and save "Radiance" variable as images
def save_radiance_as_images(nc_file_path, orbit_number):
    with Dataset(nc_file_path, 'r') as nc:
        radiance = nc.variables['Radiance'][:]
        for i in range(radiance.shape[0]):
            save_combined_image(radiance, i, output_folder, orbit_number)

# Function to parse orbit number from the file name
def parse_orbit_number(file_name):
    parts = file_name.split('_')
    return int(parts[3])

# Read and save radiance images for all .nc files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.nc'):
        nc_file_path = os.path.join(input_folder, file_name)
        orbit_number = parse_orbit_number(file_name)
        save_radiance_as_images(nc_file_path, orbit_number)
