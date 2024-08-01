# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:15:17 2024

@author: Anh
"""

import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, RangeSlider, Button
import os
import re

# Define the path to the parent directory where the dataset is located
parent_directory = 'nc_files_with_mlcloud'

# Define the orbit number
orbit_number = 90  # orbit number

# Pad the orbit number with zeros until it has 5 digits
orbit_str = str(orbit_number).zfill(5)

# Search for the correct file name in all subdirectories
pattern = re.compile(r'awe_l1c_(.*)_' + orbit_str + r'_(.*)\.nc')
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
mlcloud = dataset.variables['MLCloud'][:]
# print("=== Global Attributes ===")
# for attr in dataset.ncattrs():
#     print(f"{attr}: {dataset.getncattr(attr)}")

# print("\n=== Dimensions ===")
# for dim in dataset.dimensions.keys():
#     print(f"{dim}: {len(dataset.dimensions[dim])}")

# print("\n=== Variables ===")
# for var in dataset.variables.keys():
#     print(f"{var}: {dataset.variables[var]}")
#     print("Attributes:")
#     for attr in dataset.variables[var].ncattrs():
#         print(f"    {attr}: {dataset.variables[var].getncattr(attr)}")

# Close the dataset
dataset.close()

# Initial setup
current_time_step = 0
vmin_default = 4
vmax_default = 12
show_bounding_box = True  # Variable to track the state of the bounding box display

# Create figure and axes for the plot
fig, ax = plt.subplots(figsize=(12, 12))

# Adjust the figure to add space for the sliders
plt.subplots_adjust(bottom=0.4)

# Create the range slider for vmin and vmax on the bottom right
ax_range_slider = plt.axes([0.1, 0.2, 0.7, 0.03], facecolor='lightgoldenrodyellow')
range_slider = RangeSlider(ax_range_slider, 'vmin - vmax', 0, 30, valinit=(vmin_default, vmax_default))

# Create the slider for time step on the bottom left
ax_slider = plt.axes([0.1, 0.15, 0.7, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Time Step', 0, radiance.shape[0] - 1, valinit=current_time_step, valfmt='%0.0f')

# Create the MLCloud bar axis
ax_mlcloud_bar = plt.axes([0.1, 0.1, 0.7, 0.03], facecolor='lightgoldenrodyellow')

colorbar = None  # To keep track of the colorbar

def update_plot(time_step):
    global colorbar, current_time_step
    current_time_step = int(time_step)
    
    # Get vmin and vmax from the range slider
    vmin, vmax = range_slider.val
    
    # Ensure vmin <= vmax
    vmin, vmax = min(vmin, vmax), max(vmin, vmax)
    
    # Clear previous content
    ax.clear()
    radiance_at_time = radiance[current_time_step, :, :]
    iss_lat = iss_latitude[current_time_step]
    iss_lon = iss_longitude[current_time_step]
    cloud = mlcloud[current_time_step]
    
    # Plot the radiance data
    img = ax.imshow(radiance_at_time, origin='lower', cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(f'Radiance at Time Step {current_time_step}\nISS Position: Lat {iss_lat:.2f}, Lon {iss_lon:.2f}\n MLCloud {cloud}\n Orbit Number: {orbit_str}')
    ax.set_xlabel('Spatial Dimension X')
    ax.set_ylabel('Spatial Dimension Y')
    
    # Draw bounding box if enabled
    if show_bounding_box:
        rect = plt.Rectangle((142, -1), 16, 301, linewidth=1, edgecolor='blue', facecolor='none', linestyle='-')
        ax.add_patch(rect)
    
    # Set the aspect of the plot axis to equal, enforcing a 1:1 aspect ratio
    ax.set_aspect('equal')

    # Add colorbar if it doesn't exist
    if colorbar is None:
        colorbar = plt.colorbar(img, ax=ax, orientation='vertical')
    else:
        colorbar.update_normal(img)

    plt.draw()

def plot_mlcloud_bar():
    # Plot MLCloud bar
    colors = ['gray' if val == 0 else 'blue' for val in mlcloud]
    ax_mlcloud_bar.bar(range(len(mlcloud)), [1] * len(mlcloud), color=colors, width=1, edgecolor='none')
    ax_mlcloud_bar.set_xlim([0, len(mlcloud)])
    ax_mlcloud_bar.set_yticks([])
    ax_mlcloud_bar.set_xticks([])

def update_vmin_vmax(event):
    global current_time_step
    radiance_at_time = radiance[current_time_step, :, :]
    radiance_flat = radiance_at_time.flatten()
    
    # Remove NaN values before calculating percentiles
    radiance_flat = radiance_flat[~np.isnan(radiance_flat)]
    
    # Ensure there are no issues with empty arrays
    if len(radiance_flat) == 0:
        raise ValueError("No valid data to compute percentiles.")
    
    # Compute the 95th percentile values for vmin and vmax
    vmin = np.percentile(radiance_flat, 0.4)*0.93
    vmax = np.percentile(radiance_flat, 99.7)*1.05
    
    # Update the range slider
    range_slider.set_val((vmin, vmax))

def toggle_bounding_box(event):
    global show_bounding_box
    show_bounding_box = not show_bounding_box
    update_plot(current_time_step)

# Connect the slider and range slider to the update_plot function
slider.on_changed(lambda val: update_plot(val))
range_slider.on_changed(lambda val: update_plot(slider.val))

# Create a button to update vmin and vmax based on the 95th percentile
ax_button_vmin_vmax = plt.axes([0.38, 0.25, 0.12, 0.03], facecolor='lightgoldenrodyellow')
button_vmin_vmax = Button(ax_button_vmin_vmax, 'Set vmin-vmax (V)')

# Connect the button to the update_vmin_vmax function
button_vmin_vmax.on_clicked(update_vmin_vmax)

# Create a button to toggle the bounding box display
ax_button_bbox = plt.axes([0.52, 0.25, 0.12, 0.03], facecolor='lightgoldenrodyellow')
button_bbox = Button(ax_button_bbox, 'Toggle BBox (B)')

# Connect the button to the toggle_bounding_box function
button_bbox.on_clicked(toggle_bounding_box)

# Function to handle key presses for time step navigation
def on_key(event):
    global current_time_step

    if event.key == 'right':
        current_time_step = min(current_time_step + 1, radiance.shape[0] - 1)
    elif event.key == 'left':
        current_time_step = max(current_time_step - 1, 0)
    elif event.key == 'up':
        current_time_step = max(current_time_step - 1, 0)
    elif event.key == 'down':
        current_time_step = min(current_time_step + 1, radiance.shape[0] - 1)
    elif event.key == 'v':
        update_vmin_vmax(None)  # Call the function to update vmin and vmax
    elif event.key == 'b':
       toggle_bounding_box(None)

    slider.set_val(current_time_step)  # This will automatically update the plot via the slider's on_changed event

# Connect the key press event to the on_key function
fig.canvas.mpl_connect('key_press_event', on_key)

# Set a timer to handle continuous key presses
timer_interval = 20  # Set the interval to a lower value for more frequent updates
timer = fig.canvas.new_timer(interval=timer_interval)
timer.add_callback(lambda: on_key(None))
timer.start()

# Initial plot update
update_plot(current_time_step)
plot_mlcloud_bar()

plt.show()
