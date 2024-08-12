# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:10:28 2024

@author: Anh
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

# Ensure TensorFlow is using GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define your data augmentation pipeline
def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    # image = tf.image.random_brightness(image, max_delta=0.2)
    # image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image

# Set new image size based on bounding box dimensions
new_image_size = (100, 20)

# Load Data and Resize
data = tf.keras.utils.image_dataset_from_directory(
    'images',
    image_size=new_image_size,
    color_mode='rgb',  # Load images as RGB
    batch_size=32,
    label_mode='int'  # Ensure labels are integers for binary classification
)

# Apply augmentations
data = data.map(lambda x, y: (augment(x), y))

# Further preprocessing for ResNet-50
data = data.map(lambda x, y: (preprocess_input(x), y))

# Split Data
data_size = data.cardinality().numpy()
train_size = int(data_size * 0.7)
val_size = int(data_size * 0.2)
test_size = int(data_size * 0.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# Define Transfer Learning Model using ResNet-50
def create_resnet_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(new_image_size[0], new_image_size[1], 3)))
    base_model.trainable = False  # Freeze the base model layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)  # Assuming binary classification
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

model = create_resnet_model()

# Training and evaluating model
results = []
histories = []

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

logdir = f'logs/resnet_model'
tensorboard_callback = TensorBoard(log_dir=logdir)
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True
)

start_time = time.time()

hist = model.fit(
    train,
    epochs=200,
    validation_data=val,
    callbacks=[tensorboard_callback, early_stopping_callback]
)

end_time = time.time()
training_time = end_time - start_time

histories.append(hist)

# Evaluate
pre = tf.keras.metrics.Precision()
re = tf.keras.metrics.Recall()
acc = tf.keras.metrics.BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

precision = pre.result().numpy()
recall = re.result().numpy()
accuracy = acc.result().numpy()

results.append({
    'Model': 'ResNet50',
    'Precision': precision,
    'Recall': recall,
    'Accuracy': accuracy,
    'Training Time (s)': training_time
})

# Save the Model
os.makedirs('models', exist_ok=True)
model.save('models/DeepLearning_resnet_model_3fs.h5')

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('resnet_model_performance_comparison.csv', index=False)

print("Performance comparison saved to resnet_model_performance_comparison.csv")

# Save epoch-wise accuracy and validation accuracy
epoch_results = []

for i, hist in enumerate(histories):
    for epoch in range(len(hist.history['accuracy'])):
        epoch_results.append({
            'Model': 'ResNet50',
            'Epoch': epoch + 1,
            'Accuracy': hist.history['accuracy'][epoch],
            'Val_Accuracy': hist.history['val_accuracy'][epoch]
        })

epoch_results_df = pd.DataFrame(epoch_results)
epoch_results_df.to_csv('resnet_epoch_performance_comparison.csv', index=False)

print("Epoch performance comparison saved to resnet_epoch_performance_comparison.csv")

# Plot validation accuracy for ResNet-50 model
plt.figure(figsize=(12, 8))
for i, hist in enumerate(histories):
    plt.plot(hist.history['val_accuracy'], label=f'ResNet50 Val Accuracy')

plt.axhline(y=0.8, color='r', linestyle='--', label='80% Accuracy')
plt.axhline(y=0.9, color='g', linestyle='--', label='90% Accuracy')
plt.title('Validation Accuracy Trends for ResNet-50')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()
