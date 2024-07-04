import pandas as pd
import matplotlib.pyplot as plt

# Load the results from the CSV file
results_df = pd.read_csv('predictions_orbit_661.csv')

# Extract data for plotting
image_paths = results_df['Image Path']
predictions = results_df['Prediction']
prediction_probs = results_df['Prediction Probability']
running_avg_probs = results_df['Running Average Probability']
running_avg_binary_pred = results_df['Running Average Binary Prediction']

# Plot settings
plt.figure(figsize=(15, 12))

# Plot prediction probabilities
plt.subplot(4, 1, 1)
plt.plot(prediction_probs, label='Prediction Probability')
plt.title('Prediction Probability per Frame')
plt.xlabel('Frame Index')
plt.ylabel('Prediction Probability')
plt.legend()

# Plot running average probabilities
plt.subplot(4, 1, 2)
plt.plot(running_avg_probs, label='Running Average Probability (window size=9)', color='orange')
plt.title('Running Average Prediction Probability per Frame')
plt.xlabel('Frame Index')
plt.ylabel('Running Average Probability')
plt.legend()

# Plot binary predictions
plt.subplot(4, 1, 3)
plt.plot(predictions, label='Binary Prediction', color='green')
plt.title('Binary Predictions per Frame')
plt.xlabel('Frame Index')
plt.ylabel('Binary Prediction (0 or 1)')
plt.ylim(-0.1, 1.1)
plt.legend()

# Plot running average binary predictions
plt.subplot(4, 1, 4)
plt.plot(running_avg_binary_pred, label='Running Average Binary Prediction', color='red')
plt.title('Running Average Binary Predictions per Frame')
plt.xlabel('Frame Index')
plt.ylabel('Running Average Binary Prediction (0 or 1)')
plt.ylim(-0.1, 1.1)
plt.legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()
