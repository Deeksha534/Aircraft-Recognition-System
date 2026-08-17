import os, shutil, glob
from sklearn.model_selection import train_test_split

source_root = "dataset/auto_labels"
target_root = "detector_dataset"

images_train = os.path.join(target_root, "train/images")
labels_train = os.path.join(target_root, "train/labels")
images_val = os.path.join(target_root, "val/images")
labels_val = os.path.join(target_root, "val/labels")

os.makedirs(images_train, exist_ok=True)
os.makedirs(labels_train, exist_ok=True)
os.makedirs(images_val, exist_ok=True)
os.makedirs(labels_val, exist_ok=True)

for cls in os.listdir(source_root):
    cls_path = os.path.join(source_root, cls)
    img_dir = os.path.join(cls_path, "images")
    lbl_dir = os.path.join(cls_path, "labels")

    if not os.path.isdir(img_dir): 
        continue

    images = glob.glob(img_dir + "/*.jpg") + glob.glob(img_dir + "/*.png") + glob.glob(img_dir + "/*.jpeg")
    labels = [os.path.join(lbl_dir, os.path.splitext(os.path.basename(i))[0] + ".txt") for i in images]

    # remove images that don't have labels
    images_final = []
    labels_final = []

    for img, lbl in zip(images, labels):
        if os.path.exists(lbl):
            images_final.append(img)
            labels_final.append(lbl)

    # Split 80% train, 20% val
    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(images_final, labels_final, test_size=0.2, random_state=42)

    # Copy data
    for img, lbl in zip(train_imgs, train_lbls):
        shutil.copy(img, images_train)
        shutil.copy(lbl, labels_train)

    for img, lbl in zip(val_imgs, val_lbls):
        shutil.copy(img, images_val)
        shutil.copy(lbl, labels_val)

print("Dataset Split and Copied Successfully!")
