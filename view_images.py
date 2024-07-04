# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:15:17 2024

@author: Anh
"""

import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, RangeSlider
import cv2
import os

# Define the path to the folder where the dataset is located
dataset_folder = os.path.join('l1r_11_updated_07032024')

# Define the filename of the dataset
dataset_filename = 'awe_l1r_q20_2023326T0108_00001_v01.nc'

# Combine the folder path and filename to get the full path to the dataset
dataset_path = os.path.join(dataset_folder, dataset_filename)

# Load the dataset
dataset = nc.Dataset(dataset_path, 'r')
radiance = dataset.variables['Radiance'][:]
iss_latitude = dataset.variables['ISS_Latitude'][:]  # Load ISS latitude data
iss_longitude = dataset.variables['ISS_Longitude'][:]  # Load ISS longitude data

print("=== Global Attributes ===")
for attr in dataset.ncattrs():
    print(f"{attr}: {dataset.getncattr(attr)}")

print("\n=== Dimensions ===")
for dim in dataset.dimensions.keys():
    print(f"{dim}: {len(dataset.dimensions[dim])}")

print("\n=== Variables ===")
for var in dataset.variables.keys():
    print(f"{var}: {dataset.variables[var]}")
    print("Attributes:")
    for attr in dataset.variables[var].ncattrs():
        print(f"    {attr}: {dataset.variables[var].getncattr(attr)}")

# Close the dataset
dataset.close()

# Initial setup
current_time_step = 0
vmin_default = 4
vmax_default = 12

# Create figure and axes for the plot and optical flow
fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # 1 row, 2 columns for side-by-side

# Adjust the figure to add space for the slider and range slider
plt.subplots_adjust(bottom=0.3)

# Create the slider for time step on the bottom left
ax_slider = plt.axes([0.1, 0.05, 0.35, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Time Step', 0, radiance.shape[0] - 1, valinit=current_time_step, valfmt='%0.0f')

# Create the range slider for vmin and vmax on the bottom right
ax_range_slider = plt.axes([0.55, 0.05, 0.35, 0.03], facecolor='lightgoldenrodyellow')
range_slider = RangeSlider(ax_range_slider, 'vmin - vmax', 0, 28, valinit=(vmin_default, vmax_default))

# Function to calculate and display optical flow
def display_optical_flow(ax, prev_img, next_img):
    # Convert images to grayscale
    prev_gray = cv2.normalize(prev_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    next_gray = cv2.normalize(next_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Create an empty image to draw the flow vectors
    hsv = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    # Calculate the magnitude and angle of the flow vectors
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert HSV to RGB for visualization
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Display the optical flow
    ax.imshow(rgb_flow, origin='lower')
    ax.set_title('Optical Flow')
    ax.set_xlabel('Spatial Dimension X')
    ax.set_ylabel('Spatial Dimension Y')
    ax.set_aspect('equal')

def update_plot(time_step):
    global current_time_step
    current_time_step = int(time_step)
    
    # Get vmin and vmax from the range slider
    vmin, vmax = range_slider.val
    
    # Clear previous content
    axs[0].clear()
    radiance_at_time = radiance[current_time_step, :, :]
    iss_lat = iss_latitude[current_time_step]
    iss_lon = iss_longitude[current_time_step]
    
    # Define circle parameters
    cx, cy, r = 150, 132, 166  # Center and radius of the circle
    
    # Create a grid of x, y coordinates that match the pixel positions
    y, x = np.ogrid[:radiance_at_time.shape[0], :radiance_at_time.shape[1]]
    
    # Calculate the distance of all points from the center of the circle
    distance_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Create a mask for points outside the circle (distance from center > radius)
    outside_circle_mask = distance_from_center > r
    
    # Apply the mask to set values outside the circle to NaN
    radiance_inside_circle = np.copy(radiance_at_time)
    radiance_inside_circle[outside_circle_mask] = np.nan  # Use np.nan for missing data
    
    # Plot the masked radiance data
    axs[0].imshow(radiance_inside_circle, origin='lower', cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    axs[0].set_title(f'Radiance at Time Step {current_time_step}\nISS Position: Lat {iss_lat:.2f}, Lon {iss_lon:.2f}')
    axs[0].set_xlabel('Spatial Dimension X')
    axs[0].set_ylabel('Spatial Dimension Y')
    axs[0].set_aspect('equal')
    
    # Display the optical flow if not the first frame
    axs[1].clear()
    if current_time_step > 5:
        prev_radiance = radiance[current_time_step - 6, :, :]
        display_optical_flow(axs[1], prev_radiance, radiance_at_time)
    else:
        axs[1].set_title('Optical Flow (Not Available)')
        axs[1].set_xlabel('Spatial Dimension X')
        axs[1].set_ylabel('Spatial Dimension Y')
        axs[1].set_aspect('equal')
        axs[1].imshow(np.zeros_like(radiance_at_time), origin='lower', cmap='gray')

    plt.draw()

# Connect the slider and range slider to the update_plot function
slider.on_changed(update_plot)
range_slider.on_changed(lambda val: update_plot(slider.val))

# Function to handle key presses for time step navigation
def on_key(event):
    global current_time_step
    if event.key == 'right':
        current_time_step = min(current_time_step + 1, radiance.shape[0] - 1)
    elif event.key == 'left':
        current_time_step = max(current_time_step - 1, 0)
    elif event.key == 'up':
        current_time_step = min(current_time_step + 20, radiance.shape[0] - 1)
    elif event.key == 'down':
        current_time_step = max(current_time_step - 20, 0)
    slider.set_val(current_time_step)  # This will automatically update the plot via the slider's on_changed event

# Connect the key press event to the on_key function
fig.canvas.mpl_connect('key_press_event', on_key)

# Initial plot update
update_plot(current_time_step)

plt.show()
