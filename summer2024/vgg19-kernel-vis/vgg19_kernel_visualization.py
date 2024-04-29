# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:12:05 2024

@author: Willa

Inspiration and code from https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
Additional editing to add the values with help from ChatGPT
"""
from keras.applications.vgg19 import VGG19
from matplotlib import pyplot
import numpy as np

# load the model
model = VGG19()

# retrieve weights from the second hidden layer
filters, biases = model.layers[1].get_weights()

# Find global min and max across all filters to standardize grayscale color range
global_min = np.min(filters)
global_max = np.max(filters)

# plot first few filters
n_filters, ix = 6, 1

# Adjust these dimensions to increase the figure size
fig_size_width = 20
fig_size_height = n_filters * 3
pyplot.figure(figsize=(fig_size_width, fig_size_height))

for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, i]
    # plot each channel separately
    for j in range(3):
        # specify subplot and turn of axis
        ax = pyplot.subplot(n_filters, 3, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Standardize the grayscale range using the global min and max
        pyplot.imshow(f[:, :, j], cmap='gray', vmin=global_min, vmax=global_max)
        
        # Annotating values is optional and can be removed if it makes the plot too busy
        # Adjust fontsize as needed for readability
        fontsize = 20  # You can increase or decrease this value as needed
        for x in range(f.shape[0]):
            for y in range(f.shape[1]):
                ax.text(y, x, round(f[x, y, j], 2), ha='center', va='center', color='red', fontsize=fontsize)

        ix += 1

# Adjust layout to ensure no overlapping text and save the figure
pyplot.tight_layout()
pyplot.show()