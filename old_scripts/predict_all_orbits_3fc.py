import tensorflow as tf
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
import time

# Define the path to the saved model
model_path = 'models/DeepLearning_resnet_model.h5'

# Load the saved model
model = load_model(model_path)

# Define input and output folders
input_folder = 'images_to_predict'
output_folder = 'predictions'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# New image size based on bounding box dimensions
new_image_size = (20, 300)  # Correct dimensions to match model's input shape

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
    return int(filename.split('_')[1].replace('.png', ''))

# Function to filter binary predictions
# Only cloud if there are at least 30 consecutive cloud frames 
def filter_binary_predictions(binary_preds, min_consecutive_frames=30):
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

# Dictionary to store results for each orbit
results = {}

# Start timer
start_time = time.time()

# Iterate through images in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.png'):
        orbit_number = extract_orbit_number(file_name)
        frame_number = extract_frame_number(file_name)
        if orbit_number not in results:
            results[orbit_number] = {
                'image_paths': [],
                'frame_numbers': [],
                'predictions': [],
                'prediction_probs': [],
                'running_avg_probs': [],
                'running_avg_binary_pred': [],
                'filtered_binary_pred': []
            }
        
        image_path = os.path.join(input_folder, file_name)
        results[orbit_number]['image_paths'].append(image_path)
        results[orbit_number]['frame_numbers'].append(frame_number)
        
        # Preprocess the image
        image = preprocess_image(image_path)
        
        # Predict the class and probability
        prediction_prob = model.predict(image)[0][0]
        prediction = 1 if prediction_prob > 0.5 else 0
        
        results[orbit_number]['predictions'].append(prediction)
        results[orbit_number]['prediction_probs'].append(prediction_prob)

# End timer
end_time = time.time()
prediction_time = end_time - start_time

# Calculate running averages and binary predictions for each orbit
for orbit_number in results:
    # Sort data by 'Frame #' first
    df = pd.DataFrame({
        'Frame #': results[orbit_number]['frame_numbers'],
        'Prediction Probability': results[orbit_number]['prediction_probs']
    }).sort_values(by='Frame #')
    
    prediction_probs = df['Prediction Probability'].to_numpy()
    
    # Calculate running averages 
    running_avg_probs = []
    for i in range(len(prediction_probs)):
        start_index = max(0, i - 3)
        end_index = min(len(prediction_probs), i + 4)
        window = prediction_probs[start_index:end_index]
        running_avg = np.mean(window)
        running_avg_probs.append(running_avg)
    
    # Convert running average probabilities to binary predictions
    running_avg_binary_pred = (np.array(running_avg_probs) > 0.5).astype(int)
    
    # Apply filtering to binary predictions
    filtered_binary_pred = filter_binary_predictions(running_avg_binary_pred)
    
    # Store results in the dictionary
    results[orbit_number]['running_avg_probs'] = running_avg_probs
    results[orbit_number]['running_avg_binary_pred'] = running_avg_binary_pred
    results[orbit_number]['filtered_binary_pred'] = filtered_binary_pred
    
    # Merge with the sorted DataFrame and save to CSV
    results_df = df.copy()
    results_df['Running Average Probability'] = running_avg_probs
    results_df['Running Average Binary Prediction'] = running_avg_binary_pred
    results_df['Filtered Binary Prediction'] = filtered_binary_pred

    output_csv = os.path.join(output_folder, f'predictions_orbit_{orbit_number}.csv')
    results_df.to_csv(output_csv, index=False)

print(f"Predictions saved to {output_folder}")
print(f"Time needed to predict all orbits: {prediction_time:.2f} seconds")
