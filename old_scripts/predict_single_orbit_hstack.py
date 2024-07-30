# import tensorflow as tf
# import os
# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.resnet50 import preprocess_input
# import cv2
# import time

# # Define the path to the saved model
# model_path = 'models/DeepLearning_resnet_model.h5'

# # Load the saved model
# model = load_model(model_path)

# # Define input and output folders
# input_folder = 'images_to_predict'
# output_csv = 'predictions_orbit_661.csv'

# # Function to preprocess the image
# def preprocess_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     image = cv2.resize(image, (300, 300))
#     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#     image = np.expand_dims(image, axis=0)
#     image = preprocess_input(image)
#     return image

# # Initialize lists to store results
# image_paths = []
# predictions = []
# prediction_probs = []

# # Start timer
# start_time = time.time()

# # Iterate through images in the input folder
# for file_name in os.listdir(input_folder):
#     if file_name.endswith('.png') and 'orbit661' in file_name:
#         image_path = os.path.join(input_folder, file_name)
#         image_paths.append(image_path)
        
#         # Preprocess the image
#         image = preprocess_image(image_path)
        
#         # Predict the class and probability
#         prediction_prob = model.predict(image)[0][0]
#         prediction = 1 if prediction_prob > 0.5 else 0
        
#         predictions.append(prediction)
#         prediction_probs.append(prediction_prob)

# # End timer
# end_time = time.time()
# prediction_time = end_time - start_time

# # Calculate running average of prediction probabilities (window size 9)
# running_avg_probs = []
# window_size = 9

# for i in range(len(prediction_probs)):
#     start_index = max(0, i - 4)
#     end_index = min(len(prediction_probs), i + 5)
#     window = prediction_probs[start_index:end_index]
#     running_avg = np.mean(window)
#     running_avg_probs.append(running_avg)

# # Calculate binary prediction from running average
# running_avg_binary_pred = (np.array(running_avg_probs) > 0.5).astype(int)

# # Save results to CSV
# results_df = pd.DataFrame({
#     'Image Path': image_paths,
#     'Prediction': predictions,
#     'Prediction Probability': prediction_probs,
#     'Running Average Probability': running_avg_probs,
#     'Running Average Binary Prediction': running_avg_binary_pred
# })
# results_df.to_csv(output_csv, index=False)

# print(f"Predictions saved to {output_csv}")
# print(f"Time needed to predict one orbit: {prediction_time:.2f} seconds")

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
output_csv = 'predictions_orbit_661.csv'

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (900, 300))  # Ensure the dimensions are (900, 300)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

# Initialize lists to store results
image_paths = []
predictions = []
prediction_probs = []

# Start timer
start_time = time.time()

# Iterate through images in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.png') and 'orbit661' in file_name:
        image_path = os.path.join(input_folder, file_name)
        image_paths.append(image_path)
        
        # Preprocess the image
        image = preprocess_image(image_path)
        
        # Predict the class and probability
        prediction_prob = model.predict(image)[0][0]
        prediction = 1 if prediction_prob > 0.5 else 0
        
        predictions.append(prediction)
        prediction_probs.append(prediction_prob)

# End timer
end_time = time.time()
prediction_time = end_time - start_time

# Calculate running average of prediction probabilities (window size 9)
running_avg_probs = []
window_size = 9

for i in range(len(prediction_probs)):
    start_index = max(0, i - 4)
    end_index = min(len(prediction_probs), i + 5)
    window = prediction_probs[start_index:end_index]
    running_avg = np.mean(window)
    running_avg_probs.append(running_avg)

# Calculate binary prediction from running average
running_avg_binary_pred = (np.array(running_avg_probs) > 0.5).astype(int)

# Save results to CSV
results_df = pd.DataFrame({
    'Image Path': image_paths,
    'Prediction': predictions,
    'Prediction Probability': prediction_probs,
    'Running Average Probability': running_avg_probs,
    'Running Average Binary Prediction': running_avg_binary_pred
})
results_df.to_csv(output_csv, index=False)

print(f"Predictions saved to {output_csv}")
print(f"Time needed to predict one orbit: {prediction_time:.2f} seconds")
