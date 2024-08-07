import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from netCDF4 import Dataset
from sklearn.metrics import confusion_matrix

# Define the output folders and the CSV file path
nc_output_folder = 'nc_files_with_mlcloud'
csv_file_path = 'cloud_intervals_center_to_compare_Aug_1.csv'
predictions_folder = 'filtered_predictions'
comparison_csv_file_path = 'cloud_intervals_center_to_compare_Aug_1.csv'
figures_folder = 'Figures/3fc'

# Ensure the figures output folder exists
os.makedirs(figures_folder, exist_ok=True)

# Read the CSV file with cloud intervals
cloud_intervals_df = pd.read_csv(csv_file_path)

# Read the CSV file with orbits to compare
comparison_orbits_df = pd.read_csv(comparison_csv_file_path)
comparison_orbits = comparison_orbits_df['Orbit #'].unique()

# Function to extract orbit number from filename
def extract_orbit_number(filename):
    return int(filename.split('_')[2])

# Function to create an array with 0 and 1 based on cloud intervals
def create_cloud_array(mlcloud_length, intervals):
    cloud_array = np.zeros(mlcloud_length)
    for start, end in intervals:
        if not pd.isna(start) and not pd.isna(end):
            cloud_array[int(start):int(end) + 1] = 1
    return cloud_array

# Function to calculate and plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, orbit_number, figures_folder):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = tn + fp + fn + tp
    
    # Calculate percentages
    confusion = np.array([[tn, fp], [fn, tp]])
    percentages = confusion / total * 100
    
    # Plot confusion matrix with larger font
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=percentages, fmt='.2f', cmap='Blues', cbar=False,
                xticklabels=['Predicted No Cloud', 'Predicted Cloud'],
                yticklabels=['Actual No Cloud', 'Actual Cloud'], annot_kws={"size": 16})
    plt.title(f'Confusion Matrix for Orbit {orbit_number}', fontsize=18)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('Actual Label', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(os.path.join(figures_folder, f'confusion_matrix_orbit_{orbit_number}.png'))
    plt.close()
    
    # Print confusion matrix metrics
    accuracy = (tp + tn) / total * 100
    print(f"Orbit {orbit_number}:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"True Negatives: {tn} ({percentages[0, 0]:.2f}%)")
    print(f"False Positives: {fp} ({percentages[0, 1]:.2f}%)")
    print(f"False Negatives: {fn} ({percentages[1, 0]:.2f}%)")
    print(f"True Positives: {tp} ({percentages[1, 1]:.2f}%)")

# Read the filtered binary predictions from the predictions folder and plot
for file_name in os.listdir(predictions_folder):
    if file_name.endswith('.csv'):
        orbit_number = int(file_name.split('_')[2].replace('.csv', ''))
        
        # Only process orbits listed in the comparison CSV
        if orbit_number in comparison_orbits:
            csv_file_path = os.path.join(predictions_folder, file_name)
            
            # Filter CSV rows for the current orbit
            orbit_intervals_df = cloud_intervals_df[cloud_intervals_df['Orbit #'] == orbit_number]
            
            # Read cloud intervals
            intervals = [(row['Start'], row['End']) for _, row in orbit_intervals_df.iterrows()]
            
            # Load the filtered binary predictions
            predictions_df = pd.read_csv(csv_file_path)
            filtered_binary_pred = predictions_df['Filtered Binary Prediction'].to_numpy()
            
            # Create cloud array
            cloud_array = create_cloud_array(len(filtered_binary_pred), intervals)
            
            # Create subplots
            fig, axs = plt.subplots(2, 1, figsize=(12, 8))
            axs[0].plot(filtered_binary_pred, label='Filtered Prediction', color='green', linestyle='-')
            axs[0].set_title(f'Filtered Predictions for Orbit {orbit_number}')
            axs[0].set_xlabel('Frame #')
            axs[0].set_ylabel('Value')
            axs[0].legend()
            
            axs[1].plot(cloud_array, label='Manual Labelling', color='red', linestyle='-')
            axs[1].set_title(f'Manual Labelling for Orbit {orbit_number}')
            axs[1].set_xlabel('Frame #')
            axs[1].set_ylabel('Value')
            axs[1].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(figures_folder, f'predictions_vs_manual_orbit_{orbit_number}.png'))
            plt.close()
            
            # Plot confusion matrix
            plot_confusion_matrix(cloud_array, filtered_binary_pred, orbit_number, figures_folder)

print("Processing and plotting completed.")
