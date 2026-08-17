# scripts/train_detector.py
from ultralytics import YOLO

# use 'yolov8s.pt' for a balance. If low-memory, switch to 'yolov8n.pt'
model = YOLO("yolov8n.pt")

model.train(data="detector_dataset/data.yaml", epochs=60, imgsz=640, batch=2, name="aircraft_detector")
print("YOLO training done. Check runs/detect/aircraft_detector/weights/best.pt")
