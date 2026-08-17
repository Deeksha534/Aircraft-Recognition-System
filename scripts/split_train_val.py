import os, shutil, random
from glob import glob

random.seed(42)
src_root = "dataset/auto_labels"
dst_root = "detector_dataset"

# clean
shutil.rmtree(dst_root, ignore_errors=True)
os.makedirs(os.path.join(dst_root, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(dst_root, "train", "labels"), exist_ok=True)
os.makedirs(os.path.join(dst_root, "val", "images"), exist_ok=True)
os.makedirs(os.path.join(dst_root, "val", "labels"), exist_ok=True)

pairs = []
for combo in os.listdir(src_root):
    cdir = os.path.join(src_root, combo)
    imgs = glob(os.path.join(cdir, "images", "*.jpg"))
    for img in imgs:
        lbl = os.path.join(cdir, "labels", os.path.basename(img).replace(".jpg",".txt"))
        if os.path.exists(lbl):
            pairs.append((img,lbl))

print(f"✅ Found {len(pairs)} labeled aircraft images total")

random.shuffle(pairs)
split = int(0.8 * len(pairs))
train, val = pairs[:split], pairs[split:]

def move(split_pairs, sub):
    for img,lbl in split_pairs:
        shutil.copy(img, os.path.join(dst_root, sub, "images", os.path.basename(img)))
        shutil.copy(lbl, os.path.join(dst_root, sub, "labels", os.path.basename(lbl)))

move(train, "train")
move(val, "val")

with open(os.path.join(dst_root, "data.yaml"), "w") as f:
    f.write(
        "train: " + os.path.abspath(os.path.join(dst_root,"train","images")).replace("\\","/") + "\n" +
        "val: "   + os.path.abspath(os.path.join(dst_root,"val","images")).replace("\\","/")   + "\n\n" +
        "nc: 1\nnames: [\"aircraft\"]\n"
    )

print("🎯 Detector dataset successfully built!")
print(f"   Train images: {len(train)}")
print(f"   Val images:   {len(val)}")
