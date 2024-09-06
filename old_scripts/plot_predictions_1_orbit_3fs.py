# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:17:53 2024

@author: domin
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the path to the CSV file
csv_file_path = 'predictions/3fsx3/predictions_orbit_90.csv'

# Read the CSV file
data = pd.read_csv(csv_file_path)

# Ensure the output folder exists for the plots
output_folder = 'plots'
os.makedirs(output_folder, exist_ok=True)

# Plot each box index
for box_idx in data['Box Index'].unique():
    # Filter the data for the current box index
    box_data = data[data['Box Index'] == box_idx]

    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot prediction probabilities
    plt.plot(box_data['Frame #'], box_data['Prediction Probability'], label='Prediction Probability', color='blue')
    
    # Plot running average probabilities
    plt.plot(box_data['Frame #'], box_data['Running Average Probability'], label='Running Average Probability', color='green')
    
    # Plot filtered binary predictions
    plt.plot(box_data['Frame #'], box_data['Filtered Binary Prediction'], label='Filtered Binary Prediction', color='red', linestyle='--')

    plt.xlabel('Frame #')
    plt.ylabel('Value')
    plt.title(f'Predictions for Box Index {box_idx}')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_file_path = os.path.join(output_folder, f'predictions_box_{box_idx}.png')
    plt.savefig(plot_file_path)
    plt.close()

    print(f"Plot saved to {plot_file_path}")

print("All plots generated and saved.")
