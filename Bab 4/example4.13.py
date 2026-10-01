# Example 4.13
# Real-time Image Classification using VGG16

import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.imagenet_utils import decode_predictions


# Image size
image_size = 224

# Load VGG16 model
model = vgg16.VGG16(weights='imagenet')

print(model.summary())


# Open camera
camera = cv2.VideoCapture(0)

# Check whether camera is available
if not camera.isOpened():
    print("Error: Camera tidak dapat dibuka.")
    exit()


while camera.isOpened():

    # Read frame from camera
    ok, cam_frame = camera.read()

    if not ok:
        print("Error: Tidak dapat membaca frame kamera.")
        break

    # Resize frame for VGG16
    frame = cv2.resize(cam_frame, (image_size, image_size))

    # Convert image to array
    numpy_image = img_to_array(frame)

    # Add batch dimension
    image_batch = np.expand_dims(numpy_image, axis=0)

    # Preprocess image for VGG16
    processed_image = vgg16.preprocess_input(image_batch.copy())

    # Get predicted probabilities
    predictions = model.predict(processed_image, verbose=0)

    # Decode predictions
    label = decode_predictions(predictions, top=1)

    # Get predicted class name
    class_name = label[0][0][1]

    # Get confidence
    confidence = label[0][0][2] * 100

    # Display prediction on camera frame
    text = "VGG16: {}, {:.1f}%".format(
        class_name,
        confidence
    )

    cv2.putText(
        cam_frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # Show camera
    cv2.imshow('Video Image', cam_frame)

    # Press ESC to quit
    key = cv2.waitKey(30)

    if key == 27:
        break


# Release camera
camera.release()

# Close all OpenCV windows
cv2.destroyAllWindows()