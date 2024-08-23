import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, RangeSlider, Button
import os
import re

# Define the path to the parent directory where the dataset is located
parent_directory = 'one_nc_file_with_mlcloud'

# Define the orbit number
orbit_number = 90  # Orbit number

# Define the bounding boxes (all 40 boxes)
boxes = [
    (60, 80, 0, 100),     # Box 0
    (60, 80, 100, 200),   # Box 1
    (60, 80, 200, 300),   # Box 2
    (120, 140, 0, 100),   # Box 3
    (120, 140, 100, 200), # Box 4
    (120, 140, 200, 300), # Box 5
    (180, 200, 0, 100),   # Box 6
    (180, 200, 100, 200), # Box 7
    (180, 200, 200, 300), # Box 8
    (0, 20, 0, 100),      # Box 9
    (0, 20, 100, 200),    # Box 10
    (0, 20, 200, 300),    # Box 11
    (20, 40, 0, 100),     # Box 12
    (20, 40, 100, 200),   # Box 13
    (20, 40, 200, 300),   # Box 14
    (40, 60, 0, 100),     # Box 15
    (40, 60, 100, 200),   # Box 16
    (40, 60, 200, 300),   # Box 17
    (80, 100, 0, 100),    # Box 18
    (80, 100, 100, 200),  # Box 19
    (80, 100, 200, 300),  # Box 20
    (100, 120, 0, 100),   # Box 21
    (100, 120, 100, 200), # Box 22
    (100, 120, 200, 300), # Box 23
    (140, 160, 0, 100),   # Box 24
    (140, 160, 100, 200), # Box 25
    (140, 160, 200, 300), # Box 26
    (160, 180, 0, 100),   # Box 27
    (160, 180, 100, 200), # Box 28
    (160, 180, 200, 300), # Box 29
    (200, 220, 0, 100),   # Box 30
    (200, 220, 100, 200), # Box 31
    (200, 220, 200, 300), # Box 32
    (220, 240, 0, 100),   # Box 33
    (220, 240, 100, 200), # Box 34
    (220, 240, 200, 300), # Box 35
    (240, 260, 0, 100),   # Box 36
    (240, 260, 100, 200), # Box 37
    (240, 260, 200, 300), # Box 38
    (260, 280, 100, 200), # Box 39
]

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
mlcloud = dataset.variables['MLCloud'][:]  # Load MLCloud data

# Close the dataset
dataset.close()

# Initial setup
current_time_step = 0
show_lines = False  # Variable to track the state of the vertical and horizontal lines display
ml_threshold = 45  # Default ML threshold

# Calculate initial vmin and vmax using the 0.4th and 99.7th percentiles
radiance_at_time_0 = radiance[0, :, :]
radiance_flat = radiance_at_time_0.flatten()
radiance_flat = radiance_flat[~np.isnan(radiance_flat)]
vmin_default = np.percentile(radiance_flat, 0.4) * 0.96
vmax_default = np.percentile(radiance_flat, 99.7) * 1.05

# Create figure and axes for the plot
fig, ax = plt.subplots(figsize=(12, 12))

# Adjust the figure to add space for the sliders and information
plt.subplots_adjust(bottom=0.35)

# Create the range slider for vmin and vmax on the bottom right
ax_range_slider = plt.axes([0.1, 0.1, 0.7, 0.03], facecolor='lightgoldenrodyellow')
range_slider = RangeSlider(ax_range_slider, 'vmin - vmax', 0, 40, valinit=(vmin_default, vmax_default))

# Create the slider for time step on the bottom left
ax_slider = plt.axes([0.1, 0.05, 0.7, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Time Step', 0, radiance.shape[0] - 1, valinit=current_time_step, valfmt='%0.0f')

# Create the slider for ML threshold
ax_ml_threshold_slider = plt.axes([0.1, 0.15, 0.7, 0.03], facecolor='lightgoldenrodyellow')
ml_threshold_slider = Slider(ax_ml_threshold_slider, 'ML Threshold', 0, 100, valinit=ml_threshold, valfmt='%0.0f')

colorbar = None  # To keep track of the colorbar

# Set initial vmin and vmax
vmin, vmax = vmin_default, vmax_default

def update_plot(time_step):
    global colorbar, current_time_step, ml_threshold
    current_time_step = int(time_step)
    ml_threshold = int(ml_threshold_slider.val)
    
    # Get vmin and vmax from the range slider
    vmin, vmax = range_slider.val
    
    # Ensure vmin <= vmax
    vmin, vmax = min(vmin, vmax), max(vmin, vmax)
    
    # Clear previous content
    ax.clear()
    radiance_at_time = radiance[current_time_step, :, :]
    iss_lat = iss_latitude[current_time_step]
    iss_lon = iss_longitude[current_time_step]
    
    # Plot the radiance data
    img = ax.imshow(radiance_at_time, origin='lower', cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(f'Radiance at Time Step {current_time_step}\nISS Position: Lat {iss_lat:.2f}, Lon {iss_lon:.2f}\nOrbit Number: {orbit_str}')
    ax.set_xlabel('Spatial Dimension X')
    ax.set_ylabel('Spatial Dimension Y')
    
    # Draw vertical and horizontal lines if enabled
    if show_lines:
        for x in range(19, radiance_at_time.shape[1], 20):
            ax.axvline(x=x, color='green', linestyle='-')
        for y in range(99, radiance_at_time.shape[0], 100):
            ax.axhline(y=y, color='green', linestyle='-')
    
    # Initialize cloud center and cloud presence
    cloud_center = False
    cloud_count = 0
    
    # Highlight the boxes where MLCloud is above the threshold
    for box_idx, (x_start, x_end, y_start, y_end) in enumerate(boxes):
        if mlcloud[current_time_step, box_idx] >= ml_threshold:
            # Create a blue overlay with some transparency
            overlay = plt.Rectangle(
                (x_start, y_start), 
                x_end - x_start, 
                y_end - y_start,
                linewidth=0, 
                edgecolor='none', 
                facecolor='blue', 
                alpha=0.25
            )
            ax.add_patch(overlay)
            cloud_count += 1
            # Check if the box is one of the center boxes (24, 25, 26)
            if box_idx in [24, 25, 26]:
                cloud_center = True
    
    # Set the aspect of the plot axis to equal, enforcing a 1:1 aspect ratio
    ax.set_aspect('equal')

    # Add colorbar if it doesn't exist
    if colorbar is None:
        colorbar = plt.colorbar(img, ax=ax, orientation='vertical')
    else:
        colorbar.update_normal(img)

    # Display cloud center status and cloud presence count
    cloud_center_status = "Yes" if cloud_center else "No"
    ax.text(0.95, 0.05, f"Cloud Center: {cloud_center_status}", 
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, color='green', fontsize=12, fontweight='bold')
    ax.text(0.95, 0.02, f"Cloud Presence: {cloud_count}", #(0 best, 40 worst)
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, color='green', fontsize=12, fontweight='bold')

    plt.draw()

def update_vmin_vmax(event):
    global vmin, vmax, current_time_step
    radiance_at_time = radiance[current_time_step, :, :]
    radiance_flat = radiance_at_time.flatten()
    
    # Remove NaN values before calculating percentiles
    radiance_flat = radiance_flat[~np.isnan(radiance_flat)]
    
    # Ensure there are no issues with empty arrays
    if len(radiance_flat) == 0:
        raise ValueError("No valid data to compute percentiles.")
    
    # Compute the percentile values for vmin and vmax
    vmin = np.percentile(radiance_flat, 0.4) * 0.96
    vmax = np.percentile(radiance_flat, 99.7) * 1.05
    
    # Update the range slider
    range_slider.set_val((vmin, vmax))

def increase_range(event):
    global vmin, vmax
    vmin, vmax = range_slider.val
    range_slider.set_val((vmin * 0.96, vmax * 1.05))

def decrease_range(event):
    global vmin, vmax
    vmin, vmax = range_slider.val
    range_slider.set_val((vmin / 0.96, vmax / 1.05))

def toggle_lines(event):
    global show_lines
    show_lines = not show_lines
    update_plot(current_time_step)

# Connect the slider and range slider to the update_plot function
slider.on_changed(lambda val: update_plot(val))
range_slider.on_changed(lambda val: update_plot(slider.val))
ml_threshold_slider.on_changed(lambda val: update_plot(slider.val))

# Create a button to update vmin and vmax based on the 95th percentile
ax_button_vmin_vmax = plt.axes([0.1, 0.2, 0.14, 0.03], facecolor='lightgoldenrodyellow')
button_vmin_vmax = Button(ax_button_vmin_vmax, 'Set vmin-vmax (V)')

# Connect the button to the update_vmin_vmax function
button_vmin_vmax.on_clicked(update_vmin_vmax)

# Create a button to toggle the lines display
ax_button_lines = plt.axes([0.46, 0.2, 0.1, 0.03], facecolor='lightgoldenrodyellow')
button_lines = Button(ax_button_lines, 'Toggle Lines')

# Connect the button to the toggle_lines function
button_lines.on_clicked(toggle_lines)

# Create a button to increase the range
ax_button_increase_range = plt.axes([0.26, 0.2, 0.08, 0.03], facecolor='lightgoldenrodyellow')
button_increase_range = Button(ax_button_increase_range, '+Range (B)')

# Connect the button to the increase_range function
button_increase_range.on_clicked(increase_range)

# Create a button to decrease the range
ax_button_decrease_range = plt.axes([0.36, 0.2, 0.08, 0.03], facecolor='lightgoldenrodyellow')
button_decrease_range = Button(ax_button_decrease_range, '-Range (C)')

# Connect the button to the decrease_range function
button_decrease_range.on_clicked(decrease_range)

# Function to handle key presses for time step navigation and setting vmin-vmax
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
        update_vmin_vmax(None)
    elif event.key == 'b':
        increase_range(None)
    elif event.key == 'c':
        decrease_range(None)

    slider.set_val(current_time_step)  # This will automatically update the plot via the slider's on_changed event

# Connect the key press event to the on_key function
fig.canvas.mpl_connect('key_press_event', on_key)

# Initial plot update
update_plot(current_time_step)

plt.show()

