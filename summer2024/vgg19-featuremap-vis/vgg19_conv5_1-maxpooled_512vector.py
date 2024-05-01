# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 22:24:55 2024

@author: Willa

Further developed from my feature map outputs plots with the use of ChatGPT4. That code was originally:
    Inspired by https://neuralpixels.com/visualize-layers-in-vgg19/
    Created based on code from https://stackoverflow.com/questions/53436960/display-output-of-vgg19-layer-as-image
    With modifications from the low-resolution code from Woo et al. 2023 https://www.nature.com/articles/s41586-023-06259-2
    And input from ChatGPT4
"""

import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model

# Load and preprocess the image
img_path = 'C:/Users/Willa/Desktop/vgg19_featuremap_visualization/hist_processed.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img)
x = np.expand_dims(img, axis=0)
x = preprocess_input(x)

# Load the pre-trained VGG19 model with the default configuration
model = VGG19(weights='imagenet', include_top=False)

# Create a model that processes inputs through VGG19 and outputs the activations from block5_conv1
layer_output = model.get_layer('block5_conv1').output
activation_model = Model(inputs=model.input, outputs=layer_output)

# Get the activations for the input image
activations = activation_model.predict(x)

# Extract the max values across the spatial dimensions of each feature map
max_values = np.max(activations, axis=(1, 2)).flatten()  # Flatten to ensure it is 1D

# Print the vector of max values
print("Max values from block5_conv1 layer:", max_values)

# Save the activations to a CSV file
output_dir = 'C:/Users/Willa/Desktop/vgg19_featuremap_visualization'
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
output_path = os.path.join(output_dir, 'block5_conv1_activations.csv')
# Convert to DataFrame and transpose to fit the CSV format expected
pd.DataFrame([max_values], columns=[f"Feature_{i+1}" for i in range(len(max_values))]).to_csv(output_path, index=False)

print(f"Max values have been saved to {output_path}.")
