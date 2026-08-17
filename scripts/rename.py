import os

root = "dataset/auto_labels"

rename_map = {
    "Rafale": "Rafale",
    "F-16 Fighting Falcon": "F16",
    "F-15 Eagle": "F15",
    "MiG-29": "Mig29",
    "Su-30": "Su30",
    "Airbus A320": "A320",
    "Boeing 737": "B737",
    "ATR 72": "ATR72",
    "AH-64 Apache": "Apache",
    "UH-60 Black Hawk": "UH60",
    "MQ-9 Reaper Drone": "MQ9",
    "DJI Phantom Drone": "Phantom"
}

for old, new in rename_map.items():
    old_path = f"{root}/{old.replace(' ', '_')}"
    new_path = f"{root}/{new}"
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"✅ Renamed: {old} → {new}")
