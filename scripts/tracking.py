import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional


class TrackState:
    """Represents the real-time tracking state and history for a single aircraft."""

    def __init__(self, track_id: int, max_history: int = 40):
        self.track_id: int = track_id
        self.trajectory: deque = deque(maxlen=max_history)
        self.class_history: deque = deque(maxlen=10)
        self.last_seen_frame: int = 0
        self.smooth_label: str = "Unknown"
        self.smooth_conf: float = 0.0
        self.det_conf: float = 0.0
        self.bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)

    def update(
        self,
        bbox: Tuple[int, int, int, int],
        label: str,
        class_conf: float,
        det_conf: float,
        frame_idx: int,
    ):
        """Update track with new frame detection and classification data."""
        self.bbox = bbox
        self.det_conf = det_conf
        self.last_seen_frame = frame_idx

        # Calculate bounding box center
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        self.trajectory.append((cx, cy))

        # Record prediction in rolling history
        self.class_history.append((label, class_conf))

        # Compute smoothed label using weighted accumulated confidence
        label_scores: Dict[str, float] = {}
        label_counts: Dict[str, int] = {}
        for l, c in self.class_history:
            label_scores[l] = label_scores.get(l, 0.0) + c
            label_counts[l] = label_counts.get(l, 0) + 1

        best_label = max(label_scores, key=lambda k: label_scores[k])
        self.smooth_label = best_label
        self.smooth_conf = label_scores[best_label] / max(label_counts[best_label], 1)


class TrackManager:
    """Manages multi-object aircraft tracking lifecycle, trajectory trails, and cleanup."""

    def __init__(self, max_history: int = 40, max_stale_frames: int = 30):
        self.max_history: int = max_history
        self.max_stale_frames: int = max_stale_frames
        self.tracks: Dict[int, TrackState] = {}
        self.total_unique_tracks: int = 0

    def get_or_create(self, track_id: int) -> TrackState:
        """Retrieve existing track or initialize a new track state."""
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState(track_id, max_history=self.max_history)
            self.total_unique_tracks += 1
        return self.tracks[track_id]

    def update_track(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        label: str,
        class_conf: float,
        det_conf: float,
        frame_idx: int,
    ) -> TrackState:
        """Update a specific track and return the updated state."""
        track = self.get_or_create(track_id)
        track.update(bbox, label, class_conf, det_conf, frame_idx)
        return track

    def cleanup_stale(self, current_frame_idx: int) -> List[int]:
        """Remove tracks that have not been detected within max_stale_frames."""
        stale_ids = [
            tid
            for tid, state in self.tracks.items()
            if (current_frame_idx - state.last_seen_frame) > self.max_stale_frames
        ]
        for tid in stale_ids:
            del self.tracks[tid]
        return stale_ids

    def draw_trajectories(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (10, 132, 212),
        base_thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw smooth, fading motion trajectory trails connecting recent center positions.
        Color is specified in BGR (default tactical amber: 10, 132, 212).
        """
        for track in self.tracks.values():
            pts = list(track.trajectory)
            num_pts = len(pts)
            if num_pts < 2:
                continue

            for i in range(1, num_pts):
                pt1 = pts[i - 1]
                pt2 = pts[i]

                # Gradually increase thickness and intensity towards the current position
                fraction = i / float(num_pts)
                thickness = max(1, int(base_thickness * fraction))

                # Draw trajectory segment
                cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

                # Draw small tracking breadcrumbs
                if i % 4 == 0 or i == num_pts - 1:
                    radius = max(1, int(3 * fraction))
                    cv2.circle(frame, pt2, radius, (72, 182, 122), -1, cv2.LINE_AA)

        return frame

    def reset(self):
        """Clear all active tracks and reset counter for a new session."""
        self.tracks.clear()
        self.total_unique_tracks = 0
