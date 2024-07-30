import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from netCDF4 import Dataset
from sklearn.metrics import confusion_matrix

# Define the output folder for the new .nc files and the CSV file path
nc_output_folder = 'nc_files_with_mlcloud'
csv_file_path = 'cloud_intervals_to_compare_july_29.csv'

# Read the CSV file with cloud intervals
cloud_intervals_df = pd.read_csv(csv_file_path)

# Function to extract orbit number from filename
def extract_orbit_number(filename):
    return int(filename.split('_')[4])

# Function to create an array with 0 and 1 based on cloud intervals
def create_cloud_array(mlcloud_length, intervals):
    cloud_array = np.zeros(mlcloud_length)
    for start, end in intervals:
        if not pd.isna(start) and not pd.isna(end):
            cloud_array[int(start):int(end) + 1] = 1
    return cloud_array

# Function to calculate and plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, orbit_number):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = tn + fp + fn + tp
    
    # Calculate percentages
    confusion = np.array([[tn, fp], [fn, tp]])
    percentages = confusion / total * 100
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=percentages, fmt='.2f', cmap='Blues', cbar=False,
                xticklabels=['Predicted No Cloud', 'Predicted Cloud'],
                yticklabels=['Actual No Cloud', 'Actual Cloud'])
    plt.title(f'Confusion Matrix for Orbit {orbit_number}')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.show()
    
    # Print confusion matrix metrics
    accuracy = (tp + tn) / total * 100
    print(f"Orbit {orbit_number}:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"True Negatives: {tn} ({percentages[0, 0]:.2f}%)")
    print(f"False Positives: {fp} ({percentages[0, 1]:.2f}%)")
    print(f"False Negatives: {fn} ({percentages[1, 0]:.2f}%)")
    print(f"True Positives: {tp} ({percentages[1, 1]:.2f}%)")

# Read the new l1c .nc files and plot MLCloud values and the cloud array from CSV
for file_name in os.listdir(nc_output_folder):
    if file_name.endswith('.nc'):
        orbit_number = extract_orbit_number(file_name)
        nc_file_path = os.path.join(nc_output_folder, file_name)
        
        # Filter CSV rows for the current orbit
        orbit_intervals_df = cloud_intervals_df[cloud_intervals_df['Orbit #'] == orbit_number]
        
        # Read cloud intervals
        intervals = [(row['Start'], row['End']) for _, row in orbit_intervals_df.iterrows()]
        
        with Dataset(nc_file_path, 'r') as nc:
            # Check if 'MLCloud' variable exists
            if 'MLCloud' in nc.variables:
                mlcloud = nc.variables['MLCloud'][:]
                
                # Create cloud array
                cloud_array = create_cloud_array(len(mlcloud), intervals)
                
                # Plot MLCloud values and the cloud array from CSV
                plt.figure(figsize=(12, 6))
                plt.plot(mlcloud, label='MLCloud', color='blue')
                plt.plot(cloud_array, label='Cloud Array from CSV', color='red', linestyle='--')
                plt.title(f'MLCloud Values and Cloud Array for Orbit {orbit_number}')
                plt.xlabel('Frame #')
                plt.ylabel('Value')
                plt.legend()
                plt.show()
                
                # Plot confusion matrix
                plot_confusion_matrix(cloud_array, mlcloud, orbit_number)
            else:
                print(f"'MLCloud' variable not found in {nc_file_path}")

print("Processing and plotting completed.")
