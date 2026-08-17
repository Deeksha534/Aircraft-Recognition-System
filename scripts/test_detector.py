from ultralytics import YOLO
import glob

model = YOLO("runs/detect/ImprovedDtector/weights/best.pt")

# Test on your dataset (recursively)
images = glob.glob("dataset/raw_images/**/*.jpg", recursive=True)

print(f"Found {len(images)} images to test")

model.predict(source=images[:50], save=True, conf=0.25)  # test first 50 images
