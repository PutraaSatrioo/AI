# Example 4.17 CNN Visualize Filters
# Visualize filters from the second convolutional layer

from keras.applications.vgg19 import VGG19
from matplotlib import pyplot

# Load VGG19 model
model = VGG19()

# Summarize the model
model.summary()


# =========================================================
# Display layer numbers and names
# =========================================================

n = 0

for layer in model.layers:
    print(n, layer.name)
    n += 1


# =========================================================
# Display weights of convolutional layers
# =========================================================

n = 0

for layer in model.layers:

    if 'conv' in layer.name:

        filters, biases = layer.get_weights()

        print(
            n,
            layer.name,
            filters.shape,
            biases.shape
        )

    n += 1


# =========================================================
# Select SECOND convolutional layer
# =========================================================

n = 2

filters, biases = model.layers[n].get_weights()

s = filters.shape

print("\nSelected layer:", model.layers[n].name)
print("Color channels:", s[0])
print("Filter size:", s[1], s[2])
print("Total number of filters:", s[3])


# =========================================================
# Normalize filter values to 0-1
# =========================================================

f_min, f_max = filters.min(), filters.max()

filters = (filters - f_min) / (f_max - f_min)


# =========================================================
# Visualize first 4 filters
# =========================================================

n_filters, ix = 4, 1

pyplot.figure(figsize=(10, 10))

for i in range(n_filters):

    # Get filter
    f = filters[:, :, :, i]

    # Plot each color channel separately
    for j in range(s[0]):

        ax = pyplot.subplot(
            n_filters,
            s[0],
            ix
        )

        ax.set_xticks([])
        ax.set_yticks([])

        pyplot.imshow(
            f[j, :, :],
            cmap='gray'
        )

        ix += 1


# Show figure
pyplot.show()