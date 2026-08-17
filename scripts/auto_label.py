from ultralytics import YOLO
import os
import cv2

# Load Aircraft Detector (must be downloaded already)
model = YOLO("detectors/yolo_aircraft.pt")

# Classes we want to auto-label more (low count classes)
target_classes = [
    "Airbus_A320",
    "Boeing_737",
    "AH-64_Apache",
    "DJI_Phantom_Drone"
]

print("Auto-labeling started...\n")

for cls in target_classes:
    src_folder = os.path.join("dataset/raw_images", cls)
    dst_label_folder = os.path.join("dataset/auto_labels", cls, "labels")
    os.makedirs(dst_label_folder, exist_ok=True)

    files = os.listdir(src_folder)

    for f in files:
        img_path = os.path.join(src_folder, f)

        img = cv2.imread(img_path)
        if img is None:
            print(f" Skipped unreadable image: {img_path}")
            continue

        result = model(img_path, conf=0.12, verbose=False)[0]

        if not result.boxes:
            continue

        # Save YOLO formatted bounding boxes
        save_path = os.path.join(dst_label_folder, f.replace(".jpg", ".txt").replace(".png", ".txt"))

        with open(save_path, "w") as txt:
            for box in result.boxes.xywhn:
                x, y, w, h = box[:4]
                txt.write(f"0 {float(x):.6f} {float(y):.6f} {float(w):.6f} {float(h):.6f}\n")

    print(f"Done labeling: {cls}")

print("\nAuto-labeling complete!")
