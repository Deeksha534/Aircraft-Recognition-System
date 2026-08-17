import cv2
import numpy as np
import os
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load models
detector = YOLO("runs/detect/ImprovedDetector/weights/best.pt")
classifier = load_model("models/aircraft_classifier.h5")

# Get class names from your training dataset
class_names = sorted(os.listdir("dataset/classifier_dataset/train"))

# Paths
os.makedirs("output/frames", exist_ok=True)

# Video source: 0 for webcam, or path to a video file
VIDEO_SOURCE = 0  # Change to "input_videos/test.mp4" for testing video files

# Capture video
cap = cv2.VideoCapture(VIDEO_SOURCE)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 25

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output/real_time_output.mp4", fourcc, fps, (width, height))

frame_count = 0
saved_frames = 0

print(" Starting real-time detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # YOLO detection
    results = detector(frame, conf=0.25, verbose=False)[0]

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        try:
            resized = cv2.resize(crop, (224, 224))
            arr = image.img_to_array(resized) / 255.0
            arr = np.expand_dims(arr, axis=0)
            pred = np.argmax(classifier.predict(arr, verbose=False), axis=1)[0]
            label = class_names[pred]
        except Exception as e:
            label = "Unknown"

        # Draw box + label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save sample frames
        if saved_frames < 10 and frame_count % 10 == 0:
            save_path = f"output/frames/frame_{saved_frames+1}.jpg"
            cv2.imwrite(save_path, frame)
            saved_frames += 1

    # Write frame to output video
    out.write(frame)

    # Show window
    cv2.imshow("Aircraft Recognition - Real Time", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Finished! Output video saved to: output/real_time_output.mp4")
print(" Sample frames saved to: output/frames/")
