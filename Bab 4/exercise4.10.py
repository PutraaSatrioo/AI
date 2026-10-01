# Example 4.11
# Compare VGG16 and VGG19 for image classification

import time
import numpy as np

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input as preprocess_vgg16
)

from tensorflow.keras.applications.vgg19 import (
    VGG19,
    preprocess_input as preprocess_vgg19
)

from tensorflow.keras.applications.imagenet_utils import decode_predictions


# =========================================================
# Load image
# =========================================================
img_path = 'Elephant.jpg'

img = image.load_img(
    img_path,
    target_size=(224, 224)
)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)


# =========================================================
# VGG16
# =========================================================
print("========== VGG16 ==========")

model_vgg16 = VGG16(weights='imagenet')

x16 = preprocess_vgg16(x.copy())

start_time = time.time()

predictions16 = model_vgg16.predict(x16, verbose=0)

time16 = time.time() - start_time

results16 = decode_predictions(predictions16, top=5)

print("Top 5 Predictions:")
for result in results16[0]:
    print(
        result[1],
        "-",
        round(result[2] * 100, 2),
        "%"
    )

print("Inference Time:",
      round(time16, 4),
      "seconds")


# =========================================================
# VGG19
# =========================================================
print("\n========== VGG19 ==========")

model_vgg19 = VGG19(weights='imagenet')

x19 = preprocess_vgg19(x.copy())

start_time = time.time()

predictions19 = model_vgg19.predict(x19, verbose=0)

time19 = time.time() - start_time

results19 = decode_predictions(predictions19, top=5)

print("Top 5 Predictions:")
for result in results19[0]:
    print(
        result[1],
        "-",
        round(result[2] * 100, 2),
        "%"
    )

print("Inference Time:",
      round(time19, 4),
      "seconds")


# =========================================================
# Comparison
# =========================================================
print("\n========== COMPARISON ==========")

print("VGG16 Inference Time:",
      round(time16, 4), "seconds")

print("VGG19 Inference Time:",
      round(time19, 4), "seconds")