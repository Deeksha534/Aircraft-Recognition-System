"""
detect_video.py
Usage: python scripts/detect_video.py input_video.mp4
Reads input video, performs ByteTrack multi-object tracking + fine-grained CNN classification per frame,
draws persistent track IDs, tactical bounding boxes, labels, and motion trajectory trails,
and writes output to output/detected_<input_video>.mp4
"""

import cv2, sys, os, time, json
import numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tracking import TrackManager
from movement import get_movement_summary
from geofencing import GeofenceManager

DETECTOR_WEIGHTS = os.path.join(PROJECT_ROOT, "runs", "detect", "AircraftDetector3", "weights", "best.pt")
if not os.path.exists(DETECTOR_WEIGHTS):
    DETECTOR_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "detector.pt")

CLASSIFIER_MODEL = os.path.join(PROJECT_ROOT, "models", "aircraft_classifier.h5")
CLASSES_PATH     = os.path.join(PROJECT_ROOT, "config", "classes.json")
GEOFENCES_PATH   = os.path.join(PROJECT_ROOT, "config", "geofences.json")
CONF_THRESHOLD   = 0.25
IOU_THRESHOLD    = 0.45
CLASSIFIER_SIZE  = (224, 224)
# --------------------------------

if len(sys.argv) < 2:
    print("Usage: python scripts/detect_video.py <input_video_path>")
    sys.exit(1)

input_path = sys.argv[1]
if not os.path.isfile(input_path):
    print("Input video not found:", input_path)
    sys.exit(1)

# Load classes
if not os.path.exists(CLASSES_PATH):
    print("Error: config/classes.json not found.")
    sys.exit(1)

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

# Load models
print("Loading detector & classifier models...")
detector = YOLO(DETECTOR_WEIGHTS)
classifier = load_model(CLASSIFIER_MODEL)

expected_classes = classifier.output_shape[-1]
if len(class_names) != expected_classes:
    raise ValueError(
        f"Class count mismatch: config/classes.json defines {len(class_names)} classes, "
        f"but classifier model output shape expects {expected_classes} classes."
    )

print("Loaded classes:", class_names)

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out_dir = os.path.join(PROJECT_ROOT, "output")
os.makedirs(out_dir, exist_ok=True)
output_path = os.path.join(out_dir, f"detected_{os.path.basename(input_path)}")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

tracker = TrackManager(max_history=40, max_stale_frames=30)
geofence_mgr = GeofenceManager(config_path=GEOFENCES_PATH)
frame_idx = 0
total_det = 0
start_time = time.time()
print(f"Processing video with ByteTrack + Geofencing: {input_path} ({total_frames} frames)...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    ts_now = time.strftime("%H:%M:%S")

    # ByteTrack Multi-Object Tracking
    results = detector.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False
    )[0]

    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        det_confs = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else [1.0] * len(boxes)
        track_ids = results.boxes.id.int().cpu().tolist() if results.boxes.id is not None else list(range(1, len(boxes) + 1))

        for box, d_conf, tid in zip(boxes, det_confs, track_ids):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Classify crop
            try:
                crop_resized = cv2.resize(crop, CLASSIFIER_SIZE)
                arr = image.img_to_array(crop_resized) / 255.0
                arr = np.expand_dims(arr, axis=0)
                preds = classifier.predict(arr, verbose=0)[0]
                idx = int(np.argmax(preds))
                c_conf = float(preds[idx])
                label = class_names[idx]
            except Exception:
                label, c_conf = "Unknown", 0.0

            track_state = tracker.update_track(tid, (x1, y1, x2, y2), label, c_conf, float(d_conf), frame_idx)
            total_det += 1

            display_label = track_state.smooth_label
            display_conf = track_state.smooth_conf

            # Geofencing entry evaluation (using tracked bounding box center)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            new_alerts = geofence_mgr.update_track(
                track_id=tid,
                center_point=(cx, cy),
                label=display_label,
                frame_idx=frame_idx,
                timestamp=ts_now
            )
            for alert in new_alerts:
                print(f" [ALERT] Frame {alert.frame_idx} ({alert.timestamp}): Track #{alert.track_id:02d} ({alert.label}) ENTERED {alert.zone_name} at ({cx}, {cy})")

            # Movement Intelligence computation
            move_info = get_movement_summary(track_state.trajectory, fps=fps, window=4)
            if move_info["status"] == "MOVING":
                move_text = f"{move_info['heading']} | {move_info['speed_px_sec']:.1f} px/s"
            elif move_info["status"] == "INITIALIZING":
                move_text = "INITIALIZING"
            else:
                move_text = "STATIONARY"

            # Check if this track is currently inside any restricted zone
            active_breaches = geofence_mgr.get_active_breaches().get(tid, [])
            is_breaching = len(active_breaches) > 0
            box_color = (20, 20, 230) if is_breaching else (10, 132, 212)

            # Tactical bounding box + corner brackets
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            L = 14
            for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                cv2.line(frame, (px, py), (px + dx * L, py), box_color, 2)
                cv2.line(frame, (px, py), (px, py + dy * L), box_color, 2)

            # Tactical HUD Header Badge: TRACK #01 | Class | Conf (or RESTRICTED ALERT)
            hud_text = f"TRACK #{tid:02d} | {display_label} | {display_conf:.0%}"
            if is_breaching:
                hud_text += " [RESTRICTED]"
            (tw, th), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            bw, bh = tw + 16, th + 12
            badge_bg = (20, 20, 90) if is_breaching else (46, 76, 28)
            by1 = y1 - bh if y1 - bh >= 0 else y1
            by2 = by1 + bh
            bx1 = max(0, min(x1, w - bw))
            bx2 = min(w, bx1 + bw)
            ty = by2 - 6 if y1 - bh >= 0 else by1 + th + 6
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), badge_bg, -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 1)
            cv2.putText(frame, hud_text, (bx1 + 8, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 234, 216), 2, cv2.LINE_AA)

            # Tactical Movement Sub-badge (Image-Space kinematic readout)
            (mw, mh), _ = cv2.getTextSize(move_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.rectangle(frame, (x1, y2), (x1 + mw + 10, y2 + 18), (22, 36, 16), -1)
            cv2.rectangle(frame, (x1, y2), (x1 + mw + 10, y2 + 18), box_color, 1)
            badge_color = (245, 166, 35) if move_info["status"] == "MOVING" else (184, 170, 136)
            cv2.putText(frame, move_text, (x1 + 5, y2 + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, badge_color, 1, cv2.LINE_AA)

    # Render restricted airspace geofence zones
    active_breaches_all = geofence_mgr.get_active_breaches()
    breached_zone_ids = [zid for zlist in active_breaches_all.values() for zid in zlist]
    geofence_mgr.draw_geofences(frame, show_labels=True, active_breach_zones=breached_zone_ids)

    # Render motion trails and cleanup
    tracker.draw_trajectories(frame, color=(10, 132, 212), base_thickness=2)
    stale_ids = tracker.cleanup_stale(frame_idx)
    geofence_mgr.cleanup_stale(stale_track_ids=stale_ids)

    # HUD footer
    ts = time.strftime("%H:%M:%S")
    alert_count = len(geofence_mgr.alerts_history)
    cv2.putText(frame, f"AIRD | {ts} | F:{frame_idx:04d}/{total_frames} | TRACKS:{tracker.total_unique_tracks:02d} | ALERTS:{alert_count:02d}",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (180, 160, 80), 1, cv2.LINE_AA)

    out.write(frame)
    if frame_idx % 30 == 0:
        print(f"Processed frame {frame_idx}/{total_frames} ({tracker.total_unique_tracks} unique tracks, {alert_count} geofence alerts)")

cap.release()
out.release()
elapsed = time.time() - start_time
print(f"Finished! Output saved to {output_path}")
print(f"Processed {frame_idx} frames in {elapsed:.1f}s. Unique aircraft tracked: {tracker.total_unique_tracks}. Geofence alerts: {len(geofence_mgr.alerts_history)}.")
