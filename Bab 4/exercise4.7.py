# Example 4.9 - Comparison of CNN Configurations

import time
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical


# ============================================================
# Load MNIST dataset
# ============================================================
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape to [samples][width][height][channels]
X_train = X_train.reshape(
    (X_train.shape[0], 28, 28, 1)
).astype('float32')

X_test = X_test.reshape(
    (X_test.shape[0], 28, 28, 1)
).astype('float32')

# Normalize inputs from 0-255 to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]


# ============================================================
# Function to create CNN model
# ============================================================
def create_model(filters, kernel_size):

    model = Sequential()

    model.add(Conv2D(
        filters,
        kernel_size,
        input_shape=(28, 28, 1),
        activation='relu'
    ))

    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


# ============================================================
# MODEL 1 - Original
# 32 filters, kernel 5x5
# ============================================================
print("\n===== MODEL 1: 32 FILTERS, KERNEL 5x5 =====")

model1 = create_model(32, (5, 5))

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

print("Training Time:", round(time1, 2), "seconds")
print("Test Loss:", score1[0])
print("Test Accuracy:", score1[1])
print("CNN Error: %.2f%%" % (100 - score1[1] * 100))


# ============================================================
# MODEL 2 - 64 filters
# 64 filters, kernel 5x5
# ============================================================
print("\n===== MODEL 2: 64 FILTERS, KERNEL 5x5 =====")

model2 = create_model(64, (5, 5))

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

print("Training Time:", round(time2, 2), "seconds")
print("Test Loss:", score2[0])
print("Test Accuracy:", score2[1])
print("CNN Error: %.2f%%" % (100 - score2[1] * 100))


# ============================================================
# MODEL 3 - Kernel 3x3
# 32 filters, kernel 3x3
# ============================================================
print("\n===== MODEL 3: 32 FILTERS, KERNEL 3x3 =====")

model3 = create_model(32, (3, 3))

start_time = time.time()

history3 = model3.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=200,
    verbose=1
)

time3 = time.time() - start_time

score3 = model3.evaluate(X_test, y_test, verbose=0)

print("Training Time:", round(time3, 2), "seconds")
print("Test Loss:", score3[0])
print("Test Accuracy:", score3[1])
print("CNN Error: %.2f%%" % (100 - score3[1] * 100))


# ============================================================
# COMPARISON
# ============================================================
print("\n========== COMPARISON ==========")

print("Model 1 - 32 filters, 5x5")
print("Time    :", round(time1, 2), "seconds")
print("Accuracy:", round(score1[1] * 100, 2), "%")

print("\nModel 2 - 64 filters, 5x5")
print("Time    :", round(time2, 2), "seconds")
print("Accuracy:", round(score2[1] * 100, 2), "%")

print("\nModel 3 - 32 filters, 3x3")
print("Time    :", round(time3, 2), "seconds")
print("Accuracy:", round(score3[1] * 100, 2), "%")