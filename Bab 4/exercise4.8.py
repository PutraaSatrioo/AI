# Example 4.9 - Compare CNN before and after adding layers

import time
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical


# =========================================================
# Load MNIST dataset
# =========================================================
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape data
X_train = X_train.reshape(
    (X_train.shape[0], 28, 28, 1)
).astype('float32')

X_test = X_test.reshape(
    (X_test.shape[0], 28, 28, 1)
).astype('float32')

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode output
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]


# =========================================================
# MODEL 1 - ORIGINAL
# =========================================================
print("\n===== MODEL 1 : ORIGINAL =====")

model1 = Sequential()

model1.add(Conv2D(
    32,
    (5, 5),
    input_shape=(28, 28, 1),
    activation='relu'
))

model1.add(MaxPooling2D())
model1.add(Dropout(0.2))
model1.add(Flatten())
model1.add(Dense(128, activation='relu'))
model1.add(Dense(num_classes, activation='softmax'))

model1.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

start_time = time.time()

history1 = model1.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=200,
    verbose=1
)

time1 = time.time() - start_time

score1 = model1.evaluate(X_test, y_test, verbose=0)

print("\nOriginal Model")
print("Training Time :", round(time1, 2), "seconds")
print("Accuracy      :", round(score1[1] * 100, 2), "%")
print("CNN Error     :", round(100 - score1[1] * 100, 2), "%")


# =========================================================
# MODEL 2 - ADDED Conv2D + MaxPooling + Dropout
# =========================================================
print("\n===== MODEL 2 : MODIFIED =====")

model2 = Sequential()

# Conv2D layer 1
model2.add(Conv2D(
    32,
    (5, 5),
    input_shape=(28, 28, 1),
    activation='relu'
))

# MaxPooling layer 1
model2.add(MaxPooling2D())

# Dropout layer 1
model2.add(Dropout(0.2))

# Conv2D layer 2 - ADDED
model2.add(Conv2D(
    64,
    (3, 3),
    activation='relu'
))

# MaxPooling layer 2 - ADDED
model2.add(MaxPooling2D())

# Dropout layer 2 - ADDED
model2.add(Dropout(0.2))

# Flatten
model2.add(Flatten())

# Dense layers
model2.add(Dense(128, activation='relu'))
model2.add(Dense(num_classes, activation='softmax'))

model2.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

start_time = time.time()

history2 = model2.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=200,
    verbose=1
)

time2 = time.time() - start_time

score2 = model2.evaluate(X_test, y_test, verbose=0)

print("\nModified Model")
print("Training Time :", round(time2, 2), "seconds")
print("Accuracy      :", round(score2[1] * 100, 2), "%")
print("CNN Error     :", round(100 - score2[1] * 100, 2), "%")


# =========================================================
# MODEL COMPARISON
# =========================================================
print("\n========== COMPARISON ==========")

print("Original :",
      round(time1, 2), "sec |",
      round(score1[1] * 100, 2), "% accuracy")

print("Modified:",
      round(time2, 2), "sec |",
      round(score2[1] * 100, 2), "% accuracy")