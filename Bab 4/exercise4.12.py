# Example 4.14
# Compare ResNet50 and DenseNet121
# for webcam image classification

import cv2
import numpy as np
import time

from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications import densenet

from tensorflow.keras.applications.imagenet_utils import decode_predictions


# =========================================================
# Initialize models
# =========================================================

print("Loading ResNet50...")
model_resnet = resnet50.ResNet50(weights='imagenet')

print("Loading DenseNet121...")
model_densenet = densenet.DenseNet121(weights='imagenet')


print("========== ResNet50 ==========")
model_resnet.summary()

print("\n========== DenseNet121 ==========")
model_densenet.summary()


# =========================================================
# Open camera
# =========================================================

camera = cv2.VideoCapture(0)

image_size = 224

if not camera.isOpened():
    print("Error: Camera tidak dapat dibuka.")
    exit()


print("\nKamera berhasil dibuka.")
print("Tekan ESC untuk keluar.")


# =========================================================
# Webcam loop
# =========================================================

while camera.isOpened():

    ok, cam_frame = camera.read()

    if not ok:
        print("Error: Tidak dapat membaca frame kamera.")
        break


    # =====================================================
    # Resize webcam frame
    # =====================================================

    frame = cv2.resize(
        cam_frame,
        (image_size, image_size)
    )


    # =====================================================
    # Convert image to array
    # =====================================================

    numpy_image = img_to_array(frame)


    # Add batch dimension
    image_batch = np.expand_dims(
        numpy_image,
        axis=0
    )


    # =====================================================
    # ResNet50 prediction
    # =====================================================

    processed_resnet = resnet50.preprocess_input(
        image_batch.copy()
    )

    start_resnet = time.time()

    predictions_resnet = model_resnet.predict(
        processed_resnet,
        verbose=0
    )

    time_resnet = time.time() - start_resnet


    # Decode ResNet50 prediction
    label_resnet = decode_predictions(
        predictions_resnet,
        top=1
    )[0][0]

    class_resnet = label_resnet[1]
    confidence_resnet = label_resnet[2] * 100


    # =====================================================
    # DenseNet121 prediction
    # =====================================================

    processed_densenet = densenet.preprocess_input(
        image_batch.copy()
    )

    start_densenet = time.time()

    predictions_densenet = model_densenet.predict(
        processed_densenet,
        verbose=0
    )

    time_densenet = time.time() - start_densenet


    # Decode DenseNet121 prediction
    label_densenet = decode_predictions(
        predictions_densenet,
        top=1
    )[0][0]

    class_densenet = label_densenet[1]
    confidence_densenet = label_densenet[2] * 100


    # =====================================================
    # Display results
    # =====================================================

    text_resnet = "ResNet50: {} {:.1f}% {:.3f}s".format(
        class_resnet,
        confidence_resnet,
        time_resnet
    )

    text_densenet = "DenseNet121: {} {:.1f}% {:.3f}s".format(
        class_densenet,
        confidence_densenet,
        time_densenet
    )


    # Display ResNet50 result
    cv2.putText(
        cam_frame,
        text_resnet,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 0, 0),
        2
    )


    # Display DenseNet121 result
    cv2.putText(
        cam_frame,
        text_densenet,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    # =====================================================
    # Show webcam
    # =====================================================

    cv2.imshow(
        'ResNet50 vs DenseNet121',
        cam_frame
    )


    # =====================================================
    # Press ESC to exit
    # =====================================================

    key = cv2.waitKey(1)

    if key == 27:
        break


# =========================================================
# Release camera
# =========================================================

camera.release()
cv2.destroyAllWindows()