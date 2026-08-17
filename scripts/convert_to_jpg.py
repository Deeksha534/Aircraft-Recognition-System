from PIL import Image
import os, glob

root = "dataset/raw_images"

for cls in os.listdir(root):
    folder = os.path.join(root, cls)
    if not os.path.isdir(folder): continue

    for file in glob.glob(folder + "/*"):
        try:
            img = Image.open(file).convert("RGB")
            new = file.rsplit(".",1)[0] + ".jpg"
            img.save(new, "JPEG")
            if not file.lower().endswith(".jpg"):
                os.remove(file)
        except:
            continue
