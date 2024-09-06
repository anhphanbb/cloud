import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from netCDF4 import Dataset
from sklearn.metrics import confusion_matrix

# Define the output folder for the new .nc files and the CSV file paths
nc_output_folder = 'nc_files_with_mlcloud'
csv_files = ['cloud_intervals_to_compare_dallin.csv', 
             'cloud_intervals_to_compare_ludger.csv', 
             'cloud_intervals_to_compare_dominique.csv']
predictions_folder = 'predictions'

# Read the CSV files with cloud intervals
cloud_intervals_dfs = [pd.read_csv(csv_file) for csv_file in csv_files]

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

# Function to calculate and print confusion matrix metrics
def print_confusion_matrix_metrics(y_true, y_pred, csv_label, orbit_number):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = tn + fp + fn + tp
    
    # Calculate percentages
    confusion = np.array([[tn, fp], [fn, tp]])
    percentages = confusion / total * 100
    
    # Print confusion matrix metrics
    accuracy = (tp + tn) / total * 100
    print(f"Orbit {orbit_number} ({csv_label}):")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"True Negatives: {tn} ({percentages[0, 0]:.2f}%)")
    print(f"False Positives: {fp} ({percentages[0, 1]:.2f}%)")
    print(f"False Negatives: {fn} ({percentages[1, 0]:.2f}%)")
    print(f"True Positives: {tp} ({percentages[1, 1]:.2f}%)")
    print("")

# Read the filtered binary predictions from the predictions folder and plot
for file_name in os.listdir(predictions_folder):
    if file_name.endswith('.csv'):
        orbit_number = int(file_name.split('_')[2].replace('.csv', ''))
        
        # Load the filtered binary predictions
        csv_file_path = os.path.join(predictions_folder, file_name)
        predictions_df = pd.read_csv(csv_file_path)
        filtered_binary_pred = predictions_df['Filtered Binary Prediction'].to_numpy()
        
        # Plot MLCloud values, filtered binary predictions, and cloud arrays for each CSV file
        plt.figure(figsize=(12, 6))
        plt.plot(filtered_binary_pred, label='Filtered Binary Prediction', color='green', linestyle='-')
        
        colors = ['red', 'blue', 'purple']
        linestyles = ['-', '--', '-.']
        
        for cloud_intervals_df, csv_file, color, linestyle in zip(cloud_intervals_dfs, csv_files, colors, linestyles):
            csv_label = csv_file.split('_')[-1].replace('.csv', '')
            
            # Check if the orbit number exists in the cloud intervals CSV
            if orbit_number in cloud_intervals_df['Orbit #'].values:
                # Filter CSV rows for the current orbit
                orbit_intervals_df = cloud_intervals_df[cloud_intervals_df['Orbit #'] == orbit_number]
                
                # Read cloud intervals
                intervals = [(row['Start'], row['End']) for _, row in orbit_intervals_df.iterrows()]
                
                # Create cloud array
                cloud_array = create_cloud_array(len(filtered_binary_pred), intervals)
                
                # Plot cloud array
                plt.plot(cloud_array, label=f'Cloud Array from {csv_label}', color=color, linestyle=linestyle)
                
                # Print confusion matrix metrics
                print_confusion_matrix_metrics(cloud_array, filtered_binary_pred, csv_label, orbit_number)
        
        plt.title(f'Filtered Binary Predictions and Cloud Arrays for Orbit {orbit_number}')
        plt.xlabel('Frame #')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

print("Processing and plotting completed.")
