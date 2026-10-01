# Example 4.7 - Modified

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


# Create model
model = Sequential()

# Conv2D Layer 1
model.add(Conv2D(
    32,
    (5, 5),
    input_shape=(28, 28, 1),
    activation='relu'
))

# MaxPooling Layer 1
model.add(MaxPooling2D())

# Dropout Layer 1
model.add(Dropout(0.2))

# Conv2D Layer 2 - Added
model.add(Conv2D(
    64,
    (3, 3),
    activation='relu'
))

# MaxPooling Layer 2 - Added
model.add(MaxPooling2D())

# Dropout Layer 2 - Added
model.add(Dropout(0.2))

# Flatten
model.add(Flatten())

# Dense layers
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print(model.summary())