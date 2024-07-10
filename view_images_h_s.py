# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:15:17 2024

@author: Anh
"""

import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, RangeSlider  # Import RangeSlider
import cv2
import os
import re

# Define the path to the parent directory where the dataset is located
parent_directory = 'l1r_11_updated_07032024'

# Define the orbit number
orbit_number = 6  # Example orbit number

# Pad the orbit number with zeros until it has 5 digits
orbit_str = str(orbit_number).zfill(5)

# Search for the correct file name in all subdirectories
pattern = re.compile(r'awe_l1r_(.*)_' + orbit_str + r'_(.*)\.nc')
dataset_filename = None
dataset_path = None

for root, dirs, files in os.walk(parent_directory):
    for file in files:
        if pattern.match(file):
            dataset_filename = file
            dataset_path = os.path.join(root, file)
            break
    if dataset_filename:
        break

if dataset_filename is None:
    raise FileNotFoundError(f"No file found for orbit number {orbit_str}")

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

# Create figure and axes for the plot and histogram
fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # 1 row, 2 columns for side-by-side

# Adjust the figure to add space for the slider and range slider
plt.subplots_adjust(bottom=0.3)

# Create the slider for time step on the bottom left
ax_slider = plt.axes([0.1, 0.05, 0.35, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Time Step', 0, radiance.shape[0]-1, valinit=current_time_step, valfmt='%0.0f')

# Create the range slider for vmin and vmax on the bottom right
ax_range_slider = plt.axes([0.55, 0.05, 0.35, 0.03], facecolor='lightgoldenrodyellow')
range_slider = RangeSlider(ax_range_slider, 'vmin - vmax', 0, 28, valinit=(vmin_default, vmax_default))

colorbar = None  # To keep track of the colorbar

def update_plot(time_step):
    global colorbar, current_time_step
    current_time_step = int(time_step)
    
    # Get vmin and vmax from the range slider
    vmin, vmax = range_slider.val
    
    # Clear previous content
    axs[0].clear()
    radiance_at_time = radiance[current_time_step, :, :]
    iss_lat = iss_latitude[current_time_step]
    iss_lon = iss_longitude[current_time_step]
    
    # Plot the radiance data
    img = axs[0].imshow(radiance_at_time, origin='lower', cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    axs[0].set_title(f'Radiance at Time Step {current_time_step}\nISS Position: Lat {iss_lat:.2f}, Lon {iss_lon:.2f}\nOrbit Number: {orbit_str}')
    axs[0].set_xlabel('Spatial Dimension X')
    axs[0].set_ylabel('Spatial Dimension Y')
    
    # Draw bounding box for y = -1 to y = 300
    # Center: 147, 148, 149, 150, 151, 152
    # Expand the pixel column, 5px on each side: 142->157
    rect = plt.Rectangle((142, -1), 16, 301, linewidth=1, edgecolor='blue', facecolor='none', linestyle='-')
    axs[0].add_patch(rect)
    
    # Set the aspect of the plot axis to equal, enforcing a 1:1 aspect ratio
    axs[0].set_aspect('equal')

    # Update histogram for all data points
    radiance_flat = radiance_at_time.flatten()
    hist, bins = np.histogram(radiance_flat, bins=14, range=(0, 28))
    
    axs[1].clear()
    axs[1].bar(bins[:-1], hist, width=(bins[1] - bins[0]), color='blue', edgecolor='black')
    axs[1].set_title(f'Histogram of Radiance at Time Step {current_time_step}')
    axs[1].set_xlabel('Radiance Value')
    axs[1].set_ylabel('Frequency')
    axs[1].set_ylim(0, 60000)  # Set the y-axis limit to fix the height of the histogram
    axs[1].grid(True)

    plt.draw()

# Connect the slider and range slider to the update_plot function
slider.on_changed(lambda val: update_plot(val))
range_slider.on_changed(lambda val: update_plot(slider.val))

# Function to handle key presses for time step navigation
def on_key(event):
    global current_time_step
    if event.key == 'right':
        current_time_step = min(current_time_step + 1, radiance.shape[0] - 1)
    elif event.key == 'left':
        current_time_step = max(current_time_step - 1, 0)
    elif event.key == 'up':
        current_time_step = max(current_time_step - 20, 0)
    elif event.key == 'down':
        current_time_step = min(current_time_step + 20, radiance.shape[0] - 1)
    slider.set_val(current_time_step)  # This will automatically update the plot via the slider's on_changed event

# Connect the key press event to the on_key function
fig.canvas.mpl_connect('key_press_event', on_key)

# Initial plot update
update_plot(current_time_step)

plt.show()
