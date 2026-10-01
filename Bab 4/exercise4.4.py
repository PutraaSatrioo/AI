# Example 4.4 - LeNet-5 Keras
# Modified number of filters

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D
from keras import optimizers

model = Sequential()

# First convolutional layer
model.add(Conv2D(
    filters=8,
    kernel_size=(3, 3),
    activation='relu',
    input_shape=(32, 32, 1)
))

model.add(AveragePooling2D(pool_size=(2, 2)))

# Second convolutional layer
model.add(Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu'
))

model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# Display model architecture and parameters
model.summary()