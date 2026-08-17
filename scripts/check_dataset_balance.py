# scripts/check_dataset_balance.py
import os
from collections import defaultdict

root_dir = "dataset/raw_images"

def count_images(folder):
    c = 0
    for _, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png")):
                c += 1
    return c

counts = defaultdict(dict)
total_all = 0

for aircraft_type in os.listdir(root_dir):
    type_path = os.path.join(root_dir, aircraft_type)
    if not os.path.isdir(type_path): continue
    print(f"\nChecking: {aircraft_type}")
    type_total = 0
    for model in os.listdir(type_path):
        model_path = os.path.join(type_path, model)
        if os.path.isdir(model_path):
            c = count_images(model_path)
            counts[aircraft_type][model] = c
            type_total += c
            print(f"   ├── {model:<30} : {c} images")
    total_all += type_total
    print(f"   └── Total in {aircraft_type}: {type_total} images")

print(f"\nGrand Total: {total_all} images.")
