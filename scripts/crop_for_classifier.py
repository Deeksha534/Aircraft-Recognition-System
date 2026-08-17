# scripts/crop_for_classifier.py
from ultralytics import YOLO
import cv2, os, tqdm

detector_path = "runs/detect/ImprovedDetector/weights/best.pt"
source_root = "dataset/raw_images"
output_root = "dataset/classifier_images"

os.makedirs(output_root, exist_ok=True)
model = YOLO(detector_path)

print("Scanning images...")
all_images = []
for root, _, files in os.walk(source_root):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            all_images.append(os.path.join(root, f))

for img_path in tqdm.tqdm(all_images, desc="Cropping Aircraft"):
    try:
        results = model(img_path, conf=0.25, verbose=False)[0]
        if not results.boxes:
            continue

        category = os.path.basename(os.path.dirname(img_path))
        save_folder = os.path.join(output_root, category)
        os.makedirs(save_folder, exist_ok=True)

        image = cv2.imread(img_path)
        for i, b in enumerate(results.boxes.xyxy):
            x1, y1, x2, y2 = map(int, b)
            crop = image[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(save_folder, f"{os.path.basename(img_path).split('.')[0]}_crop{i}.jpg"), crop)
    except Exception as e:
        print("⚠️ Error:", e)

print("Cropping complete! Check dataset/classifier_images/")
