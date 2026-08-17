from ddgs import DDGS
import requests, os, time, random

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
aircraft_classes = {
    "Rafale": 200,
    "F-16 Fighting Falcon": 200,
    "F-15 Eagle": 200,
    "MiG-29": 200,
    "Su-30": 200,
    "Airbus A320": 200,
    "Boeing 737": 200,
    "ATR 72": 200,
    "AH-64 Apache": 200,
    "UH-60 Black Hawk": 200,
    "MQ-9 Reaper Drone": 200,
    "DJI Phantom Drone": 200,
}

save_root = "dataset/raw_images"
os.makedirs(save_root, exist_ok=True)
error_log_path = "download_errors.log"

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def log_error(msg):
    with open(error_log_path, "a") as f:
        f.write(msg + "\n")

def safe_download(url, path, retries=3):
    """Download file with retry logic."""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(path, "wb") as f:
                    f.write(response.content)
                return True
        except Exception as e:
            time.sleep(2 ** attempt)  # exponential backoff
    log_error(f"Failed to download {url}")
    return False

# ---------------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------------
with DDGS() as ddgs:
    for name, count in aircraft_classes.items():
        folder = os.path.join(save_root, name.replace(" ", "_"))
        os.makedirs(folder, exist_ok=True)

        print(f"\nDownloading images for {name}...")

        try:
            results = list(ddgs.images(name, max_results=count))
        except Exception as e:
            log_error(f"Search failed for {name}: {e}")
            print(f"Search failed for {name}")
            continue

        for i, r in enumerate(results):
            file_path = os.path.join(folder, f"{name.replace(' ', '_')}_{i}.jpg")
            if os.path.exists(file_path):
                # 
                continue

            try:
                img_url = r["image"]
                ok = safe_download(img_url, file_path)
                if not ok:
                    continue

                # delay between downloads
                time.sleep(random.uniform(0.4, 1.3))

            except Exception as e:
                log_error(f"Error {name} -> {e}")
                continue

        print(f"Finished: {name}")
        # sleep between categories
        time.sleep(random.uniform(5, 10))

print("\nAll downloads complete! Check dataset/raw_images/")
print(f"Errors logged in: {error_log_path}")
