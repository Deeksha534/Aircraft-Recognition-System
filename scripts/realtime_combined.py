import cv2
import numpy as np
import os, sys, json, time

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

DETECTOR_PATH = os.path.join(PROJECT_ROOT, "runs", "detect", "AircraftDetector3", "weights", "best.pt")
if not os.path.exists(DETECTOR_PATH):
    DETECTOR_PATH = os.path.join(PROJECT_ROOT, "models", "detector.pt")

CLASSIFIER_PATH = os.path.join(PROJECT_ROOT, "models", "aircraft_classifier.h5")
CLASSES_PATH    = os.path.join(PROJECT_ROOT, "config", "classes.json")
GEOFENCES_PATH  = os.path.join(PROJECT_ROOT, "config", "geofences.json")
# --------------------------------

# Load class taxonomy
if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(f"Configuration file not found: {CLASSES_PATH}")
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

# Load models
print("Loading detector & classifier models...")
detector   = YOLO(DETECTOR_PATH)
classifier = load_model(CLASSIFIER_PATH)

expected_classes = classifier.output_shape[-1]
if len(class_names) != expected_classes:
    raise ValueError(
        f"Class count mismatch: config/classes.json defines {len(class_names)} classes, "
        f"but classifier model output shape expects {expected_classes} classes."
    )

# Paths
os.makedirs(os.path.join(PROJECT_ROOT, "output", "frames"), exist_ok=True)

# Video source: 0 for webcam, or path to a video file
VIDEO_SOURCE = 0

cap = cv2.VideoCapture(VIDEO_SOURCE)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 25

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video_path = os.path.join(PROJECT_ROOT, "output", "real_time_output.mp4")
out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

tracker = TrackManager(max_history=40, max_stale_frames=30)
geofence_mgr = GeofenceManager(config_path=GEOFENCES_PATH)
frame_count = 0
saved_frames = 0

print(" Starting real-time ByteTrack aircraft tracking with Geofencing... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    ts_now = time.strftime("%H:%M:%S")

    # YOLO detection + ByteTrack
    results = detector.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.25,
        iou=0.45,
        verbose=False
    )[0]

    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        det_confs = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else [1.0] * len(boxes)
        track_ids = results.boxes.id.int().cpu().tolist() if results.boxes.id is not None else list(range(1, len(boxes) + 1))

        for box, d_conf, tid in zip(boxes, det_confs, track_ids):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            try:
                resized = cv2.resize(crop, (224, 224))
                arr = image.img_to_array(resized) / 255.0
                arr = np.expand_dims(arr, axis=0)
                preds = classifier.predict(arr, verbose=False)[0]
                idx = int(np.argmax(preds))
                label = class_names[idx]
                c_conf = float(preds[idx])
            except Exception:
                label, c_conf = "Unknown", 0.0

            track_state = tracker.update_track(tid, (x1, y1, x2, y2), label, c_conf, float(d_conf), frame_count)

            # Geofencing entry evaluation
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            entry_alerts = geofence_mgr.update_track(
                track_id=tid,
                center_point=(cx, cy),
                label=track_state.smooth_label,
                frame_idx=frame_count,
                timestamp=ts_now
            )
            for alert in entry_alerts:
                print(f" [ALERT] Frame {alert.frame_idx} ({alert.timestamp}): Track #{alert.track_id:02d} ({alert.label}) ENTERED {alert.zone_name}!")

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

            # Draw tactical HUD box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            # Corner brackets
            L = 14
            for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                cv2.line(frame, (px, py), (px + dx * L, py), box_color, 2)
                cv2.line(frame, (px, py), (px, py + dy * L), box_color, 2)

            txt = f"TRACK #{tid:02d} | {track_state.smooth_label} | {track_state.smooth_conf:.0%}"
            if is_breaching:
                txt += " [RESTRICTED]"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            bw, bh = tw + 16, th + 12
            badge_bg = (20, 20, 90) if is_breaching else (46, 76, 28)
            by1 = y1 - bh if y1 - bh >= 0 else y1
            by2 = by1 + bh
            bx1 = max(0, min(x1, width - bw))
            bx2 = min(width, bx1 + bw)
            ty = by2 - 6 if y1 - bh >= 0 else by1 + th + 6
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), badge_bg, -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 1)
            cv2.putText(frame, txt, (bx1 + 8, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 234, 216), 2, cv2.LINE_AA)

            # Tactical Movement Sub-badge (Image-Space kinematic readout)
            (mw, mh), _ = cv2.getTextSize(move_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.rectangle(frame, (x1, y2), (x1 + mw + 10, y2 + 18), (22, 36, 16), -1)
            cv2.rectangle(frame, (x1, y2), (x1 + mw + 10, y2 + 18), box_color, 1)
            badge_color = (245, 166, 35) if move_info["status"] == "MOVING" else (184, 170, 136)
            cv2.putText(frame, move_text, (x1 + 5, y2 + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, badge_color, 1, cv2.LINE_AA)

    # Render restricted airspace geofences
    active_breaches_all = geofence_mgr.get_active_breaches()
    breached_zone_ids = [zid for zl in active_breaches_all.values() for zid in zl]
    geofence_mgr.draw_geofences(frame, show_labels=True, active_breach_zones=breached_zone_ids)

    # Draw trajectories
    tracker.draw_trajectories(frame, color=(10, 132, 212), base_thickness=2)
    stale_ids = tracker.cleanup_stale(frame_count)
    geofence_mgr.cleanup_stale(stale_track_ids=stale_ids)

    # Save sample frames
    if saved_frames < 10 and frame_count % 10 == 0:
        save_path = os.path.join(PROJECT_ROOT, "output", "frames", f"frame_{saved_frames+1}.jpg")
        cv2.imwrite(save_path, frame)
        saved_frames += 1

    out.write(frame)
    cv2.imshow("AIRD - Real Time Aircraft Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Finished! Output video saved to: {out_video_path}")
print(f"Total unique aircraft tracked: {tracker.total_unique_tracks}")
print(f"Total restricted-zone entry alerts logged: {len(geofence_mgr.alerts_history)}")
