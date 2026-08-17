from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np, cv2, os

detector = YOLO("runs/detect/ImprovedDetector/weights/best.pt")
classifier = load_model("models/aircraft_classifier.h5")

class_names = sorted(os.listdir("dataset/classifier_dataset/train"))

input_folder = "images_for_test_recognize"
output_folder = "output"
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
        pred = np.argmax(classifier.predict(arr), axis=1)[0]
        label = class_names[pred]

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image_bgr, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imwrite(os.path.join(output_folder, file), image_bgr)
    print(f"Processed and saved: {file}")

print(" Recognition complete! Check the 'output' folder.")
