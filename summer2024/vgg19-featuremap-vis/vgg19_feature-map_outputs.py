# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:21:34 2024

@author: Willa

Inspired by https://neuralpixels.com/visualize-layers-in-vgg19/
Created based on code from https://stackoverflow.com/questions/53436960/display-output-of-vgg19-layer-as-image
With modifications from the low-resolution code from Woo et al. 2023 https://www.nature.com/articles/s41586-023-06259-2
And input from ChatGPT4
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model


#Normally the image would be loaded in and segmented from the background using detectron2, but for the sake of keeping this
#code short and not having to import all of the models, the cuttlefish was segmented using https://segment-anything.com/demo,
#I used Pixlr Editor online to center the masked cuttlefish and make sure the resulting prepared png had a mask half the
#length of the total image size, as would be the case if the cuttlefish picture was passed through detectron2 and prepocessing.
#I then manually processed through this commented out code. The background of the grayscale image was removed so that 
#it was still a png, then passed through histogram equalization (you CANNOT do this correctly if it is not a png).
#Finally,  I filled in the background of the histogram-equalized mask with middle grey.
#When I call the image in this code, it has already passed through these steps.

#def rgb2gray(rgb):
    #gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    #cv2.imwrite(os.path.join(output_dir, 'grayscale.jpg'), gray)
    #return gray

#def histN(img):
    #gray_img = rgb2gray(img)
    #equalized_img = cv2.equalizeHist(gray_img.astype(np.uint8))
    #colorized_equalized_img = cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR)
    #cv2.imwrite(os.path.join(output_dir, 'histogram_equalized.jpg'), colorized_equalized_img)
    #return colorized_equalized_img

# Load and preprocess the image
img_path = 'C:/Users/Willa/Desktop/vgg19_featuremap_visualization/hist_processed.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img)
#img = histN(img)
x = np.expand_dims(img, axis=0)
x = preprocess_input(x)

# Load the pre-trained VGG19 model
model = VGG19(weights='imagenet', include_top=False)

# Layers for which to visualize activations
layers = ['block1_conv1', 'block1_conv2', 'block1_pool', 'block2_conv1', 
          'block2_conv2', 'block2_pool', 'block3_conv1', 'block3_conv2', 'block3_conv3', 
          'block3_conv4', 'block3_pool', 'block4_conv1', 'block4_conv2', 'block4_conv3', 
          'block4_conv4', 'block4_pool', 'block5_conv1']

# Specify output directory
output_dir = 'C:/Users/Willa/Desktop/vgg19_featuremap_visualization'
os.makedirs(output_dir, exist_ok=True)  # Create directory if it does not exist

# Iterate through each specified layer
for layer_name in layers:
    # Extract the output of the desired layer
    layer_output = model.get_layer(layer_name).output

    # Create a model that outputs the layer's activations
    activation_model = Model(inputs=model.input, outputs=layer_output)

    # Get the activations for the input image
    activations = activation_model.predict(x)

    # Determine the layout for the composite image
    num_channels = activations.shape[-1]
    rows = int(np.ceil(np.sqrt(num_channels)))
    cols = int(np.ceil(num_channels / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(rows * 2, cols * 2))

    for i, ax in enumerate(axes.flat):
        if i < num_channels:
            # Normalize and display each activation channel
            ax.imshow(activations[0, :, :, i], cmap='viridis')
            ax.axis('off')
        else:
            ax.axis('off')

    # Adjust layout and save the image
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{layer_name}_composite.jpg"), bbox_inches='tight')
    plt.close(fig)  # Close the plot to free up memory