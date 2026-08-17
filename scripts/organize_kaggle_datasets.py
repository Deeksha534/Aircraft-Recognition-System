import os
import shutil
from PIL import Image

# Where Kaggle datasets were extracted
KAGGLE_SRC = "dataset/kaggle_downloads"

# Final unified dataset folder
TARGET = "dataset/raw_images"

# Known model name mappings
MODEL_MAP = {
    # Fighter Jets
    "f16": ("Fighter_Jet", "F-16_Fighting_Falcon"),
    "rafale": ("Fighter_Jet", "Dassault_Rafale"),
    "mig29": ("Fighter_Jet", "MiG-29_Fulcrum"),
    "mig-29": ("Fighter_Jet", "MiG-29_Fulcrum"),
    "hornet": ("Fighter_Jet", "F-A-18_Super_Hornet"),
    "typhoon": ("Fighter_Jet", "Eurofighter_Typhoon"),
    "f15": ("Fighter_Jet", "F-15_Eagle"),
    "su27": ("Fighter_Jet", "Su-27_Flanker"),
    "gripen": ("Fighter_Jet", "Gripen_JAS_39"),
    "mirage": ("Fighter_Jet", "Mirage_2000"),
    "f35": ("Fighter_Jet", "F-35_Lightning_II"),

    # Commercial
    "boeing737": ("Commercial", "Boeing_737"),
    "boeing747": ("Commercial", "Boeing_747"),
    "boeing777": ("Commercial", "Boeing_777"),
    "boeing787": ("Commercial", "Boeing_787_Dreamliner"),
    "a320": ("Commercial", "Airbus_A320"),
    "a330": ("Commercial", "Airbus_A330"),
    "a350": ("Commercial", "Airbus_A350"),
    "embraer": ("Commercial", "Embraer_E190"),
    "crj": ("Commercial", "Bombardier_CRJ900"),
    "atr": ("Commercial", "ATR_72"),

    # Helicopters
    "apache": ("Helicopter", "AH-64_Apache"),
    "blackhawk": ("Helicopter", "UH-60_Black_Hawk"),
    "chinook": ("Helicopter", "CH-47_Chinook"),
    "hind": ("Helicopter", "Mi-24_Hind"),
    "bell206": ("Helicopter", "Bell_206_JetRanger"),
    "bell407": ("Helicopter", "Bell_407"),
    "ec135": ("Helicopter", "Eurocopter_EC135"),
    "aw139": ("Helicopter", "AgustaWestland_AW139"),
    "ka52": ("Helicopter", "Ka-52_Alligator"),
    "r44": ("Helicopter", "Robinson_R44"),

    # Drones / UAV
    "mq9": ("Drone", "MQ-9_Reaper"),
    "mq1": ("Drone", "MQ-1_Predator"),
    "phantom": ("Drone", "DJI_Phantom_4"),
    "mavic": ("Drone", "DJI_Mavic_3"),
    "inspire": ("Drone", "DJI_Inspire_2"),
    "anafi": ("Drone", "Parrot_Anafi"),
    "skydio": ("Drone", "Skydio_2"),
    "tb2": ("Drone", "Bayraktar_TB2"),
    "heron": ("Drone", "Heron_UAV"),
    "wingloong": ("Drone", "Wing_Loong_II"),
}

def is_image(file):
    try:
        Image.open(file).verify()
        return True
    except:
        return False

for root, dirs, files in os.walk(KAGGLE_SRC):
    for file in files:
        filepath = os.path.join(root, file)

        # Skip non-images
        if not (file.lower().endswith(".jpg") or file.lower().endswith(".png")):
            continue

        # Match filename keywords
        name = file.lower().replace(" ", "").replace("_", "")
        matched = False

        for key, (category, model) in MODEL_MAP.items():
            if key in name:
                target_folder = os.path.join(TARGET, category, model)
                os.makedirs(target_folder, exist_ok=True)

                new_name = f"{model}_{len(os.listdir(target_folder))+1}.jpg"
                shutil.copy(filepath, os.path.join(target_folder, new_name))
                matched = True
                break
        
        if not matched:
            pass  # If model not recognized, skip (safe)

print("Dataset successfully organized into dataset/raw_images/")
