import os, shutil, random

source_root = "dataset/classifier_images"
target_root = "dataset/classifier_dataset"

# Clean target if re-running
if os.path.exists(target_root):
    shutil.rmtree(target_root)

for split in ["train", "val"]:
    for cls in os.listdir(source_root):
        os.makedirs(os.path.join(target_root, split, cls), exist_ok=True)

split_ratio = 0.8

for cls in os.listdir(source_root):
    cls_path = os.path.join(source_root, cls)
    images = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)

    split_idx = int(len(images) * split_ratio)
    train_imgs, val_imgs = images[:split_idx], images[split_idx:]

    for img in train_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(target_root, "train", cls, img))
    for img in val_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(target_root, "val", cls, img))

print("✅ Classifier dataset prepared successfully!")
