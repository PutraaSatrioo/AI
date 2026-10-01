# Example 4.13
# Compare VGG16 and VGG19 for Webcam Classification

import cv2
import numpy as np
import time

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications import vgg19
from tensorflow.keras.applications.imagenet_utils import decode_predictions


# =========================================================
# Settings
# =========================================================
image_size = 224


# =========================================================
# Load VGG16 and VGG19
# =========================================================
model16 = vgg16.VGG16(weights='imagenet')
model19 = vgg19.VGG19(weights='imagenet')

print("VGG16")
model16.summary()

print("\nVGG19")
model19.summary()


# =========================================================
# Open webcam
# =========================================================
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Camera tidak dapat dibuka.")
    exit()


while camera.isOpened():

    # Read webcam frame
    ok, cam_frame = camera.read()

    if not ok:
        print("Error: Tidak dapat membaca frame.")
        break

    # Resize image
    frame = cv2.resize(
        cam_frame,
        (image_size, image_size)
    )

    # Convert image to array
    numpy_image = img_to_array(frame)

    # Add batch dimension
    image_batch = np.expand_dims(numpy_image, axis=0)


    # =====================================================
    # VGG16 Prediction
    # =====================================================
    processed16 = vgg16.preprocess_input(
        image_batch.copy()
    )

    start16 = time.time()

    prediction16 = model16.predict(
        processed16,
        verbose=0
    )

    time16 = time.time() - start16

    label16 = decode_predictions(
        prediction16,
        top=1
    )[0][0]

    class16 = label16[1]
    confidence16 = label16[2] * 100


    # =====================================================
    # VGG19 Prediction
    # =====================================================
    processed19 = vgg19.preprocess_input(
        image_batch.copy()
    )

    start19 = time.time()

    prediction19 = model19.predict(
        processed19,
        verbose=0
    )

    time19 = time.time() - start19

    label19 = decode_predictions(
        prediction19,
        top=1
    )[0][0]

    class19 = label19[1]
    confidence19 = label19[2] * 100


    # =====================================================
    # Display result
    # =====================================================

    text16 = "VGG16: {} {:.1f}% | {:.3f}s".format(
        class16,
        confidence16,
        time16
    )

    text19 = "VGG19: {} {:.1f}% | {:.3f}s".format(
        class19,
        confidence19,
        time19
    )

    cv2.putText(
        cam_frame,
        text16,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.putText(
        cam_frame,
        text19,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Show webcam
    cv2.imshow(
        'VGG16 vs VGG19',
        cam_frame
    )


    # Press ESC to quit
    key = cv2.waitKey(1)

    if key == 27:
        break


# =========================================================
# Release resources
# =========================================================
camera.release()
cv2.destroyAllWindows()