# Example 4.15
# Select model using if-else statement

import cv2
import numpy as np

from keras.applications.imagenet_utils import decode_predictions
from classification_models.keras import Classifiers


# =========================================================
# Select model
# =========================================================

model_name = input(
    "Pilih model (vgg16/resnet50/mobilenetv2/densenet201/inceptionv3): "
).lower()


# =========================================================
# Select classifier using if-elif-else
# =========================================================

if model_name == 'vgg16':

    clf, preprocess_input = Classifiers.get('vgg16')
    image_size = 224

elif model_name == 'resnet50':

    clf, preprocess_input = Classifiers.get('resnet50')
    image_size = 224

elif model_name == 'mobilenetv2':

    clf, preprocess_input = Classifiers.get('mobilenetv2')
    image_size = 224

elif model_name == 'densenet201':

    clf, preprocess_input = Classifiers.get('densenet201')
    image_size = 224

elif model_name == 'inceptionv3':

    clf, preprocess_input = Classifiers.get('inceptionv3')
    image_size = 299

else:

    print("Model tidak tersedia.")
    exit()


# =========================================================
# Create model
# =========================================================

model = clf(
    input_shape=(image_size, image_size, 3),
    weights='imagenet',
    classes=1000
)

print("\nModel yang digunakan:", model_name)
print("Image size:", image_size)

model.summary()


# =========================================================
# Open camera
# =========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Camera tidak dapat dibuka.")
    exit()


# =========================================================
# Webcam classification
# =========================================================

while True:

    ret, cam_frame = camera.read()

    if not ret:
        print("Error: Tidak dapat membaca frame kamera.")
        break

    # Resize image
    frame = cv2.resize(
        cam_frame,
        (image_size, image_size)
    )

    # Convert image to NumPy array
    image = np.asarray(frame)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Preprocess image
    image = preprocess_input(image)

    # Prediction
    preds = model.predict(
        image,
        verbose=0
    )

    # Decode predictions
    label = decode_predictions(
        preds,
        top=1
    )

    class_name = label[0][0][1]
    confidence = label[0][0][2] * 100

    # Display prediction
    text = "{}: {:.1f}%".format(
        class_name,
        confidence
    )

    cv2.putText(
        cam_frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        cam_frame,
        "Model: " + model_name,
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.imshow(
        "Classification",
        cam_frame
    )

    # Press ESC to quit
    key = cv2.waitKey(30)

    if key == 27:
        break


# =========================================================
# Release camera
# =========================================================

camera.release()
cv2.destroyAllWindows()