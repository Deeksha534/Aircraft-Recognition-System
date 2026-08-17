import os
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# =========================================================
#  PART 1: DETECTOR (YOLO)
# =========================================================
print("Evaluating YOLOv8 Detector...")

detector_model_path = "runs/detect/ImprovedDetector/weights/best.pt"
data_yaml_path = "detector_dataset/data.yaml"

detector = YOLO(detector_model_path)

# Run YOLO validation
metrics = detector.val(data=data_yaml_path, split="val")

print("\nDetector Results:")
print(f"  • Precision: {metrics.results_dict['metrics/precision(B)']:.3f}")
print(f"  • Recall:    {metrics.results_dict['metrics/recall(B)']:.3f}")
print(f"  • mAP@0.5:   {metrics.results_dict['metrics/mAP50(B)']:.3f}")
print(f"  • mAP@0.5:0.95: {metrics.results_dict['metrics/mAP50-95(B)']:.3f}")

# =========================================================
#  PART 2: CLASSIFIER (Keras)
# =========================================================
print("\nEvaluating Aircraft Classifier...")

classifier_model_path = "models/aircraft_classifier.h5"
val_dir = "dataset/classifier_dataset/val"

model = load_model(classifier_model_path)

datagen = ImageDataGenerator(rescale=1./255)
val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Evaluate
loss, acc = model.evaluate(val_gen, verbose=0)
print(f"\nClassifier Accuracy: {acc*100:.2f}%")
print(f"Classifier Loss: {loss:.4f}")

# Confusion matrix & classification report
print("\nGenerating confusion matrix...")
Y_pred = model.predict(val_gen, verbose=0)
y_pred = np.argmax(Y_pred, axis=1)
class_names = list(val_gen.class_indices.keys())

print("\nClassification Report:\n")
print(classification_report(val_gen.classes, y_pred, target_names=class_names))

cm = confusion_matrix(val_gen.classes, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Aircraft Model Classification Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# =========================================================
#  PART 3: SUMMARY
# =========================================================
print("\nFINAL SYSTEM SUMMARY:")
print("───────────────────────────")
print(f"Detector (YOLOv8): mAP@0.5 = {metrics.results_dict['metrics/mAP50(B)']:.3f}, Precision = {metrics.results_dict['metrics/precision(B)']:.3f}")
print(f"Classifier (Keras): Accuracy = {acc*100:.2f}%")
print("───────────────────────────")

# Rough Combined Metric (System Accuracy = Detector Precision × Classifier Accuracy)
system_acc = metrics.results_dict['metrics/precision(B)'] * acc
print(f"Estimated End-to-End System Accuracy: {system_acc*100:.2f}%")
