# Example - Visualize Feature Maps from VGG19

import matplotlib.pyplot as plt

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from numpy import expand_dims


# =========================================================
# Function to plot all feature maps
# =========================================================

def plot_feature_maps(feature_maps):

    # Number of columns
    col = 8

    # Number of rows
    row = int(feature_maps.shape[3] / col)

    ix = 1

    plt.figure(figsize=(20, 20))

    for _ in range(row):

        for _ in range(col):

            # Specify subplot and turn off axis
            ax = plt.subplot(row, col, ix)

            ax.set_xticks([])
            ax.set_yticks([])

            # Plot feature map in grayscale
            plt.imshow(
                feature_maps[0, :, :, ix - 1],
                cmap='gray'
            )

            ix += 1

    plt.show()


# =========================================================
# Load VGG19
# =========================================================

base_model = VGG19()

base_model.summary()


# =========================================================
# Select convolutional layer
# =========================================================

# Change this value:
# 2 = block1_conv2
# 3 = block1_conv3
# 4 = block1_conv4

n = 2


# Redefine model to output feature maps
model = Model(
    inputs=base_model.inputs,
    outputs=base_model.layers[n].output
)

print("\nSelected layer:")
print(base_model.layers[n].name)

model.summary()


# =========================================================
# Load image
# =========================================================

img = load_img(
    'Elephant.jpg',
    target_size=(224, 224)
)


# Convert image to array
img = img_to_array(img)


# Add batch dimension
img = expand_dims(
    img,
    axis=0
)


# Preprocess image
img = preprocess_input(img)


# =========================================================
# Generate feature maps
# =========================================================

feature_maps = model.predict(
    img,
    verbose=0
)

print(
    "Feature maps shape:",
    feature_maps.shape
)


# =========================================================
# Plot all feature maps
# =========================================================

plot_feature_maps(feature_maps)