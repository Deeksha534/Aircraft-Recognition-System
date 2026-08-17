import cv2, os, glob

root = "dataset/raw_images"

for cls in os.listdir(root):
    folder = os.path.join(root, cls)
    if not os.path.isdir(folder): continue

    for img_path in glob.glob(folder + "/*"):
        img = cv2.imread(img_path)
        if img is None:
            print("❌ Removing corrupt:", img_path)
            os.remove(img_path)
