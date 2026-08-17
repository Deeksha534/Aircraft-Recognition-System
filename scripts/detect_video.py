"""
detect_video.py
Usage: python scripts/detect_video.py input_video.mp4
This reads input video, performs detection + classification per frame,
draws boxes and labels, and writes output to outputs/detected_<input_video>.mp4
"""

import cv2, sys, os, time
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ---------- CONFIG ----------
DETECTOR_WEIGHTS = "runs/detect/ImprovedDetector/weights/best.pt"
CLASSIFIER_MODEL = "models/aircraft_classifier.h5"
CONF_THRESHOLD = 0.35
CLASSIFIER_SIZE = (224, 224)
# ----------------------------

if len(sys.argv) < 2:
    print("Usage: python scripts/detect_video.py <input_video_path>")
    sys.exit(1)

input_path = sys.argv[1]
if not os.path.isfile(input_path):
    print("Input video not found:", input_path)
    sys.exit(1)

# load models
detector = YOLO(DETECTOR_WEIGHTS)
classifier = load_model(CLASSIFIER_MODEL)

train_root = "dataset/classifier_dataset/train"
class_names = sorted([d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))])
print("Classes:", class_names)

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

os.makedirs("outputs", exist_ok=True)
output_path = os.path.join("outputs", f"detected_{os.path.basename(input_path)}")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

frame_idx = 0
start = time.time()
print("Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    results = detector(frame, conf=CONF_THRESHOLD, verbose=False)[0]

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # classify
            try:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                img = cv2.resize(crop_rgb, CLASSIFIER_SIZE)
                arr = image.img_to_array(img) / 255.0
                arr = np.expand_dims(arr, axis=0)
                preds = classifier.predict(arr, verbose=0)
                idx = int(np.argmax(preds, axis=1)[0])
                c_conf = float(np.max(preds))
                label = class_names[idx]
            except Exception:
                label, c_conf = "unknown", 0.0

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {c_conf:.2f}", (x1, max(15,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    out.write(frame)

cap.release()
out.release()
elapsed = time.time() - start
print(f"Saved output video to {output_path} (processed {frame_idx} frames in {elapsed:.1f}s).")
