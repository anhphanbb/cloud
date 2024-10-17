import tensorflow as tf
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
import time
import multiprocessing as mp  # Import multiprocessing
from tqdm import tqdm  # Import tqdm for progress tracking

# Define the path to the saved model
model_path = 'models/DeepLearning_resnet_model_3fs.h5'

# Load the saved model once
model = load_model(model_path)

# Define input and output folders
main_folder = 'images_to_predict_3'  # Main directory containing subfolders
output_folder = 'predictions/3fsx3'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# New image size based on bounding box dimensions
new_image_size = (20, 100)  # Correct dimensions to match model's input shape

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read image in RGB
    image = cv2.resize(image, new_image_size)  # Resize correctly
    image = preprocess_input(image)  # Use preprocess_input for RGB images
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to extract orbit number from filename
def extract_orbit_number(filename):
    return filename.split('_')[0].replace('orbit', '')

# Function to extract frame number from filename
def extract_frame_number(filename):
    return int(filename.split('_')[2].replace('.png', ''))

# Function to extract box index from filename
def extract_box_index(filename):
    return int(filename.split('_')[1].replace('box', ''))

# Function to filter binary predictions
def filter_binary_predictions(binary_preds, min_consecutive_frames=5):
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

# Function to process each subfolder
def process_subfolder(subfolder):
    subfolder_path = os.path.join(main_folder, subfolder)
    if not os.path.isdir(subfolder_path):
        return
    
    # Dictionary to store results for each orbit and each box within the current subfolder
    results = {}

    # Process images in the current subfolder
    for file_name in os.listdir(subfolder_path):
        if file_name.endswith('.png'):
            orbit_number = extract_orbit_number(file_name)
            frame_number = extract_frame_number(file_name)
            box_index = extract_box_index(file_name)
            
            if orbit_number not in results:
                results[orbit_number] = {
                    'image_paths': [],
                    'frame_numbers': [],
                    'box_indices': [],
                    'predictions': [],
                    'prediction_probs': [],
                    'running_avg_probs': [],
                    'running_avg_binary_pred': [],
                    'filtered_binary_pred': []
                }
            
            image_path = os.path.join(subfolder_path, file_name)
            results[orbit_number]['image_paths'].append(image_path)
            results[orbit_number]['frame_numbers'].append(frame_number)
            results[orbit_number]['box_indices'].append(box_index)
            
            # Preprocess the image
            image = preprocess_image(image_path)
            
            # Predict the class and probability
            prediction_prob = model.predict(image)[0][0]
            prediction = 1 if prediction_prob > 0.5 else 0
            
            results[orbit_number]['predictions'].append(prediction)
            results[orbit_number]['prediction_probs'].append(prediction_prob)

    # Calculate running averages and binary predictions for each orbit and each box in the current subfolder
    for orbit_number in results:
        # Create DataFrame for current orbit
        df = pd.DataFrame({
            'Frame #': results[orbit_number]['frame_numbers'],
            'Box Index': results[orbit_number]['box_indices'],
            'Prediction Probability': results[orbit_number]['prediction_probs']
        }).sort_values(by=['Box Index', 'Frame #'])
        
        # Iterate through each box (0 to 8 for 9 boxes)
        for box_idx in range(9):
            box_df = df[df['Box Index'] == box_idx]
            prediction_probs = box_df['Prediction Probability'].to_numpy()
            
            # Calculate running averages of 5
            running_avg_probs = []
            for i in range(len(prediction_probs)):
                start_index = max(0, i - 2)
                end_index = min(len(prediction_probs), i + 3)
                window = prediction_probs[start_index:end_index]
                running_avg = np.mean(window)
                running_avg_probs.append(running_avg)
            
            # Convert running average probabilities to binary predictions
            running_avg_binary_pred = (np.array(running_avg_probs) > 0.5).astype(int)
            
            # Apply filtering to binary predictions
            filtered_binary_pred = filter_binary_predictions(running_avg_binary_pred)
            
            # Store results in the DataFrame using .loc[]
            df.loc[df['Box Index'] == box_idx, 'Running Average Probability'] = running_avg_probs
            df.loc[df['Box Index'] == box_idx, 'Running Average Binary Prediction'] = running_avg_binary_pred
            df.loc[df['Box Index'] == box_idx, 'Filtered Binary Prediction'] = filtered_binary_pred

        # Save the results for the current orbit in the current subfolder
        output_csv = os.path.join(output_folder, f'predictions_orbit_{orbit_number}.csv')
        df.to_csv(output_csv, index=False)

# Start timer
start_time = time.time()

# Get all subfolders to process
subfolders = [subfolder for subfolder in sorted(os.listdir(main_folder)) if os.path.isdir(os.path.join(main_folder, subfolder))]

# Use multiprocessing to process subfolders in parallel with progress bar
with mp.Pool(mp.cpu_count()) as pool:
    for _ in tqdm(pool.imap_unordered(process_subfolder, subfolders), total=len(subfolders), desc="Processing Subfolders"):
        pass

# End timer
end_time = time.time()
prediction_time = end_time - start_time

print(f"Predictions saved to {output_folder}")
print(f"Time needed to predict all subfolders: {prediction_time:.2f} seconds")
