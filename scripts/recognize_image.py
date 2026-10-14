from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np, cv2, os, json

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DETECTOR_PATH = os.path.join(PROJECT_ROOT, "runs", "detect", "AircraftDetector3", "weights", "best.pt")
if not os.path.exists(DETECTOR_PATH):
    DETECTOR_PATH = os.path.join(PROJECT_ROOT, "models", "detector.pt")

CLASSIFIER_PATH = os.path.join(PROJECT_ROOT, "models", "aircraft_classifier.h5")
CLASSES_PATH    = os.path.join(PROJECT_ROOT, "config", "classes.json")

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

detector = YOLO(DETECTOR_PATH)
classifier = load_model(CLASSIFIER_PATH)

expected_classes = classifier.output_shape[-1]
if len(class_names) != expected_classes:
    raise ValueError(
        f"Class count mismatch: config/classes.json defines {len(class_names)} classes, "
        f"but classifier model output shape expects {expected_classes} classes."
    )

input_folder = os.path.join(PROJECT_ROOT, "images_for_test_recognize")
output_folder = os.path.join(PROJECT_ROOT, "output")
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(input_folder, file)
    results = detector(img_path, conf=0.25, verbose=False)[0]
    image_bgr = cv2.imread(img_path)

    if not results.boxes:
        print(f"No aircraft detected in {file}")
        continue

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        resized = cv2.resize(crop, (224, 224))
        arr = image.img_to_array(resized) / 255.0
        arr = np.expand_dims(arr, axis=0)
        preds = classifier.predict(arr, verbose=False)[0]
        idx = int(np.argmax(preds))
        label = class_names[idx]
        conf = float(preds[idx])

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (10, 132, 212), 2)
        H_img, W_img = image_bgr.shape[:2]
        txt = f"{label} | {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
        bw, bh = tw + 16, th + 12
        by1 = y1 - bh if y1 - bh >= 0 else y1
        by2 = by1 + bh
        bx1 = max(0, min(x1, W_img - bw))
        bx2 = min(W_img, bx1 + bw)
        ty = by2 - 6 if y1 - bh >= 0 else by1 + th + 6
        cv2.rectangle(image_bgr, (bx1, by1), (bx2, by2), (46, 76, 28), -1)
        cv2.rectangle(image_bgr, (bx1, by1), (bx2, by2), (10, 132, 212), 1)
        cv2.putText(image_bgr, txt, (bx1 + 8, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (240, 234, 216), 2, cv2.LINE_AA)

    cv2.imwrite(os.path.join(output_folder, file), image_bgr)
    print(f"Processed and saved: {file}")

print("Recognition complete! Check the 'output' folder.")
