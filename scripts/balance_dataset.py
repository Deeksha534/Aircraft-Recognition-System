# scripts/balance_dataset.py
import os, random, shutil

root_dir = "dataset/raw_images"

def list_images(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

def duplicate_images(folder, needed, image_files):
    if not image_files:
        print(f"Skipping duplication for {folder} — no images found.")
        return
    for i in range(needed):
        src = os.path.join(folder, random.choice(image_files))
        dest = os.path.join(folder, f"dup_{i}_{os.path.basename(src)}")
        shutil.copy(src, dest)

def trim_images(folder, excess, image_files):
    if not image_files:
        print(f"Skipping trim for {folder} — no images found.")
        return
    if excess >= len(image_files):
        print(f"Skipping trim for {folder} — not enough images to remove safely.")
        return
    remove_files = random.sample(image_files, excess)
    for f in remove_files:
        os.remove(os.path.join(folder, f))

if __name__ == "__main__":
    all_counts = []
    folders = []

    for t in os.listdir(root_dir):
        tpath = os.path.join(root_dir, t)
        if not os.path.isdir(tpath):
            continue
        for m in os.listdir(tpath):
            fpath = os.path.join(tpath, m)
            if os.path.isdir(fpath):
                count = len(list_images(fpath))
                all_counts.append(count)
                folders.append((fpath, count))

    if not all_counts:
        print("No images found. Exiting.")
        exit()

    target = int(sum(all_counts) / len(all_counts))
    print(f"Target per model: {target}")

    for folder, count in folders:
        imgs = list_images(folder)
        if count < target:
            duplicate_images(folder, target - count, imgs)
            print(f" Duplicated {target - count} images in {folder}")
        elif count > target:
            trim_images(folder, count - target, imgs)
            print(f"Trimmed {count - target} images in {folder}")
        else:
            print(f" Balanced: {folder}")

    print("Dataset balancing finished successfully.")
