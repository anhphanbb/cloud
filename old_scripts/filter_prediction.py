import os
import numpy as np
import pandas as pd

# Define input and output folders
predictions_folder = 'predictions'
output_folder = 'filtered_predictions'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to extract orbit number from filename
def extract_orbit_number(filename):
    return int(filename.split('_')[2].replace('.csv', ''))

# Function to filter binary predictions
# Only cloud if there are at least 4 consecutive cloud frames 
def filter_binary_predictions(binary_preds, min_consecutive_frames=15):
    filtered_preds = np.copy(binary_preds)
    n = len(binary_preds)
    i = 0
    while i < n:
        if binary_preds[i] == 1:
            start = i
            while i < n and binary_preds[i] == 1:
                i += 1
            length = i - start
            if length < min_consecutive_frames:
                filtered_preds[start:i] = 0
        else:
            i += 1
    return filtered_preds

# Iterate through prediction CSV files
for file_name in os.listdir(predictions_folder):
    if file_name.endswith('.csv'):
        try:
            orbit_number = extract_orbit_number(file_name)
        except ValueError:
            print(f"Skipping file with unexpected format: {file_name}")
            continue
        
        csv_file_path = os.path.join(predictions_folder, file_name)
        
        # Read the CSV file
        predictions_df = pd.read_csv(csv_file_path)
        
        # Apply filtering to binary predictions
        running_avg_binary_pred = predictions_df['Running Average Binary Prediction'].to_numpy()
        filtered_binary_pred = filter_binary_predictions(running_avg_binary_pred)
        
        # Update the DataFrame with the new filtered predictions
        predictions_df['Filtered Binary Prediction'] = filtered_binary_pred
        
        # Save the updated DataFrame to a new CSV file
        output_csv = os.path.join(output_folder, file_name)
        predictions_df.to_csv(output_csv, index=False)

print(f"Filtered predictions saved to {output_folder}")
