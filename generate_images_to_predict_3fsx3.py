import os
from netCDF4 import Dataset
import cv2
import numpy as np

# Define input and output folders
input_folder = '45'
output_folder = 'images_to_predict'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

boxes = [
    (60, 80, 0, 100),
    (60, 80, 100, 200),
    (60, 80, 200, 300),
    (120, 140, 0, 100),
    (120, 140, 100, 200),
    (120, 140, 200, 300),
    (180, 200, 0, 100),
    (180, 200, 100, 200),
    (180, 200, 200, 300),
]

# Function to save a single image for a given bounding box
def save_image(data, orbit_folder, orbit_number, frame_index, box_idx):
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
    
    # Create a three-layer image
    three_layer_image = np.zeros((data.shape[1], data.shape[2], 3), dtype=np.uint8)
    if prev_frame_norm is not None:
        three_layer_image[..., 0] = prev_frame_norm
    three_layer_image[..., 1] = norm_radiance
    if next_frame_norm is not None:
        three_layer_image[..., 2] = next_frame_norm
    
    # Get the bounding box coordinates
    x_start, x_end, y_start, y_end = boxes[box_idx]

    # Crop the image to the bounding box
    cropped_image = three_layer_image[y_start:y_end, x_start:x_end]

    # Save the image in the orbit-specific folder
    file_path = os.path.join(orbit_folder, f"orbit{orbit_number}_box{box_idx}_{frame_index}.png")
    cv2.imwrite(file_path, cropped_image)
    print(f"Saved {file_path}")

# Function to read and save "Radiance" variable as images
def save_radiance_as_images(nc_file_path, orbit_number):
    # Create a folder for the current orbit inside the output folder
    orbit_folder = os.path.join(output_folder, f'orbit_{orbit_number}')
    os.makedirs(orbit_folder, exist_ok=True)
    
    with Dataset(nc_file_path, 'r') as nc:
        radiance = nc.variables['Radiance'][:]
        for i in range(5, radiance.shape[0] - 5):
            for box_idx in range(len(boxes)):
                save_image(radiance, orbit_folder, orbit_number, i, box_idx)

# Function to parse orbit number from the file name
def parse_orbit_number(file_name):
    parts = file_name.split('_')
    return int(parts[4])

# Read and save radiance images for all .nc files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.nc'):
        nc_file_path = os.path.join(input_folder, file_name)
        orbit_number = parse_orbit_number(file_name)
        save_radiance_as_images(nc_file_path, orbit_number)
