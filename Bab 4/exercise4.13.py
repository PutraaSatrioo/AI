# Example 4.15
# Compare different image classification models
# using webcam

import cv2
import numpy as np
import time
import gc

from keras.applications.imagenet_utils import decode_predictions
from classification_models.keras import Classifiers
from keras import backend as K


# ============================================================
# Function to run one model
# ============================================================

def run_model(model_name, image_size, test_frames=30):

    print("\n========================================")
    print("MODEL:", model_name)
    print("IMAGE SIZE:", image_size)
    print("========================================")

    # Get classifier and preprocessing function
    clf, preprocess_input = Classifiers.get(model_name)

    # Create model
    model = clf(
        input_shape=(image_size, image_size, 3),
        weights='imagenet',
        classes=1000
    )

    model.summary()

    # Open webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Camera tidak dapat dibuka.")
        return None

    inference_times = []
    last_label = "Waiting..."
    last_confidence = 0

    frame_count = 0

    while camera.isOpened():

        ret, cam_frame = camera.read()

        if not ret:
            print("Error: Tidak dapat membaca frame.")
            break

        # Resize frame
        frame = cv2.resize(
            cam_frame,
            (image_size, image_size)
        )

        # OpenCV gives BGR, convert to RGB
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert to numpy array
        image = np.asarray(frame_rgb)

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        # Preprocess
        image = preprocess_input(image)

        # Measure inference time
        start_time = time.time()

        preds = model.predict(
            image,
            verbose=0
        )

        inference_time = time.time() - start_time

        inference_times.append(inference_time)

        # Decode prediction
        label = decode_predictions(
            preds,
            top=1
        )

        last_label = label[0][0][1]
        last_confidence = label[0][0][2] * 100

        # Calculate FPS
        fps = 1 / inference_time if inference_time > 0 else 0

        # Display result
        text = "{}: {:.1f}% | {:.3f}s | {:.1f} FPS".format(
            last_label,
            last_confidence,
            inference_time,
            fps
        )

        cv2.putText(
            cam_frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            cam_frame,
            "Model: {}".format(model_name),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        cv2.putText(
            cam_frame,
            "Press ESC to finish this model",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        # Show webcam
        cv2.imshow(
            "Classification - " + model_name,
            cam_frame
        )

        frame_count += 1

        # Stop after required frames
        if frame_count >= test_frames:
            break

        # ESC to stop early
        key = cv2.waitKey(1)

        if key == 27:
            break

    # Calculate average inference time
    average_time = np.mean(inference_times)

    average_fps = 1 / average_time

    print("\nResult:", model_name)
    print("Average inference time:",
          round(average_time, 4),
          "seconds")

    print("Average FPS:",
          round(average_fps, 2))

    print("Last prediction:",
          last_label)

    print("Last confidence:",
          round(last_confidence, 2),
          "%")

    # Close camera
    camera.release()
    cv2.destroyAllWindows()

    # Clear model from memory
    del model
    K.clear_session()
    gc.collect()

    return {
        "model": model_name,
        "time": average_time,
        "fps": average_fps,
        "label": last_label,
        "confidence": last_confidence
    }


# ============================================================
# Models to compare
# ============================================================

models = [
    ("vgg16", 224),
    ("resnet50", 224),
    ("mobilenetv2", 224),
    ("densenet201", 224),
    ("inceptionv3", 299)
]


# ============================================================
# Run comparison
# ============================================================

results = []

for model_name, image_size in models:

    result = run_model(
        model_name,
        image_size,
        test_frames=30
    )

    if result is not None:
        results.append(result)


# ============================================================
# Final comparison
# ============================================================

print("\n\n========================================")
print("FINAL PERFORMANCE COMPARISON")
print("========================================")

print(
    "{:<15} {:<15} {:<12} {:<15}".format(
        "Model",
        "Avg Time",
        "FPS",
        "Confidence"
    )
)

print("-" * 60)

for result in results:

    print(
        "{:<15} {:<15} {:<12} {:<15}".format(
            result["model"],
            str(round(result["time"], 4)) + " s",
            round(result["fps"], 2),
            str(round(result["confidence"], 2)) + "%"
        )
    )