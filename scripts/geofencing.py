"""
geofencing.py
Tactical Geofencing and Restricted-Zone Entry Alert Module for AIRD.

IMPORTANT NOTE:
All coordinates, boundaries, and center points in this module are measured
strictly in 2D IMAGE-SPACE (pixel coordinates). They represent screen-space
surveillance regions relative to the video or sensor frame, and do NOT reflect
real-world GPS / geographic coordinates or air-traffic radar boundaries.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def validate_polygon_coords(polygon: Any) -> Tuple[bool, str]:
    """
    Validates polygon coordinates ensuring:
    1. It is a sequence of at least 3 points.
    2. Each point is a pair of numeric coordinates [x, y].
    3. The polygon has non-zero geometric area.
    """
    if not isinstance(polygon, (list, tuple)):
        return False, "Polygon must be a list or tuple of coordinate pairs."

    if len(polygon) < 3:
        return False, f"Polygon requires at least 3 vertices, got {len(polygon)}."

    cleaned_pts = []
    for i, pt in enumerate(polygon):
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            return False, f"Vertex at index {i} must be a 2D coordinate [x, y], got {pt}."
        try:
            x, y = float(pt[0]), float(pt[1])
            cleaned_pts.append((x, y))
        except (ValueError, TypeError):
            return False, f"Vertex at index {i} contains non-numeric coordinates: {pt}."

    # Validate non-zero area
    poly_arr = np.array(cleaned_pts, dtype=np.float32)
    area = cv2.contourArea(poly_arr)
    if area <= 0.0:
        return False, "Polygon vertices define degenerate or zero-area geometry."

    return True, ""


def point_in_polygon(
    point: Tuple[Union[int, float], Union[int, float]],
    polygon: Sequence[Tuple[Union[int, float], Union[int, float]]],
) -> bool:
    """
    Determines whether a 2D point (x, y) lies inside or on the boundary of a polygon.
    Treats the polygon boundary consistently as INSIDE.

    Args:
        point: (x, y) 2D pixel coordinates.
        polygon: Sequence of (x, y) vertex coordinates.

    Returns:
        True if the point is inside or on the boundary of the polygon; False otherwise.
    """
    if not polygon or len(polygon) < 3:
        return False

    px, py = float(point[0]), float(point[1])
    poly_arr = np.array(polygon, dtype=np.float32)

    # cv2.pointPolygonTest returns:
    #   +1.0 if inside
    #    0.0 if on vertex or edge (boundary)
    #   -1.0 if outside
    dist = cv2.pointPolygonTest(poly_arr, (px, py), measureDist=False)
    return dist >= 0.0


@dataclass
class GeofenceZone:
    """Represents a single restricted airspace zone in 2D image coordinates."""

    id: str
    name: str
    polygon: List[Tuple[float, float]]
    enabled: bool = True
    color_bgr: Tuple[int, int, int] = (0, 0, 220)  # Default: Tactical Red
    description: str = ""

    def contains_point(self, point: Tuple[Union[int, float], Union[int, float]]) -> bool:
        """Check if point is inside or on boundary of this zone."""
        if not self.enabled:
            return False
        return point_in_polygon(point, self.polygon)


@dataclass
class GeofenceAlert:
    """Structured data container for a restricted-zone entry event."""

    track_id: int
    label: str
    zone_id: str
    zone_name: str
    event_type: str = "ZONE_ENTRY"
    frame_idx: int = 0
    timestamp: str = ""
    position: Tuple[int, int] = (0, 0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to serializable dictionary."""
        return {
            "track_id": self.track_id,
            "label": self.label,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "event_type": self.event_type,
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "position": list(self.position),
        }


class GeofenceManager:
    """
    Manages restricted airspace zones, target state tracking, and entry-only alerts.

    Key operational rules:
    - Entry-only alerts: Emitted strictly when an aircraft transitions from outside to inside.
    - No repeated alerts while an aircraft remains inside a zone.
    - Re-entering after exiting triggers a new entry alert.
    - Tracks first observed inside a zone do NOT generate false entry alerts.
    - Each track and zone combination is tracked independently.
    - Disabled zones produce no alerts.
    - Stale tracks are cleanly purged from state memory.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        zones: Optional[List[GeofenceZone]] = None,
    ):
        self.zones: List[GeofenceZone] = []
        # track_states maps track_id -> {zone_id: is_inside (bool)}
        self.track_states: Dict[int, Dict[str, bool]] = {}
        # Complete historical log of entry alerts emitted during this session
        self.alerts_history: List[GeofenceAlert] = []

        if zones is not None:
            for z in zones:
                self.add_zone(z)
        elif config_path is not None:
            self.load_config(config_path)

    def add_zone(self, zone: GeofenceZone) -> bool:
        """Validates and registers a new GeofenceZone."""
        is_valid, err_msg = validate_polygon_coords(zone.polygon)
        if not is_valid:
            logger.warning("Failed to add zone '%s' (%s): %s", zone.name, zone.id, err_msg)
            return False

        # Format polygon vertices as float tuples
        zone.polygon = [(float(pt[0]), float(pt[1])) for pt in zone.polygon]

        # Prevent duplicate zone IDs
        for idx, existing in enumerate(self.zones):
            if existing.id == zone.id:
                self.zones[idx] = zone
                return True

        self.zones.append(zone)
        return True

    def load_config(self, config_path: str) -> bool:
        """
        Loads and validates restricted zones from a JSON configuration file.
        Safely ignores invalid zones without halting execution.
        """
        if not os.path.exists(config_path):
            logger.warning("Geofence configuration file not found: %s", config_path)
            return False

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error("Failed to parse geofence configuration JSON '%s': %e", config_path, e)
            return False

        if not isinstance(data, dict):
            logger.warning("Geofence configuration must contain a root JSON object.")
            return False

        raw_zones = data.get("zones", [])
        if not isinstance(raw_zones, list):
            logger.warning("'zones' entry in configuration must be a list.")
            return False

        loaded_count = 0
        for entry in raw_zones:
            if not isinstance(entry, dict):
                continue

            zone_id = str(entry.get("id", f"zone_{len(self.zones) + 1}"))
            zone_name = str(entry.get("name", zone_id))
            enabled = bool(entry.get("enabled", True))
            raw_poly = entry.get("polygon", [])
            color_bgr = entry.get("color_bgr", [0, 0, 220])
            desc = str(entry.get("description", ""))

            if isinstance(color_bgr, list) and len(color_bgr) == 3:
                color_tuple = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
            else:
                color_tuple = (0, 0, 220)

            is_valid, err_msg = validate_polygon_coords(raw_poly)
            if not is_valid:
                logger.warning(
                    "Skipping invalid zone configuration '%s' (ID: %s): %s",
                    zone_name,
                    zone_id,
                    err_msg,
                )
                continue

            poly_pts = [(float(p[0]), float(p[1])) for p in raw_poly]
            zone_obj = GeofenceZone(
                id=zone_id,
                name=zone_name,
                polygon=poly_pts,
                enabled=enabled,
                color_bgr=color_tuple,
                description=desc,
            )
            if self.add_zone(zone_obj):
                loaded_count += 1

        logger.info("Successfully loaded %d geofence zone(s) from %s", loaded_count, config_path)
        return loaded_count > 0

    def update_track(
        self,
        track_id: int,
        center_point: Tuple[Union[int, float], Union[int, float]],
        label: str = "Unknown",
        frame_idx: int = 0,
        timestamp: Optional[str] = None,
    ) -> List[GeofenceAlert]:
        """
        Evaluates a single tracked aircraft against all enabled geofences.

        Emits entry alerts strictly upon an OUTSIDE -> INSIDE transition.
        Does NOT emit alerts if:
        - The track was first observed inside a zone without a known previous outside state.
        - The track remains inside the zone.
        - The track leaves the zone.
        - The zone is disabled.

        Args:
            track_id: Unique ByteTrack integer identifier.
            center_point: (cx, cy) 2D pixel center of the aircraft bounding box.
            label: Classified aircraft name/class.
            frame_idx: Current video or stream frame index.
            timestamp: Formatted time string (defaults to current time if None).

        Returns:
            List of newly generated GeofenceAlert objects for this frame step.
        """
        if timestamp is None:
            timestamp = time.strftime("%H:%M:%S")

        new_alerts: List[GeofenceAlert] = []

        if track_id not in self.track_states:
            self.track_states[track_id] = {}

        for zone in self.zones:
            if not zone.enabled:
                continue

            is_inside = zone.contains_point(center_point)
            prev_inside = self.track_states[track_id].get(zone.id, None)

            if prev_inside is None:
                # First observation of this aircraft for this specific zone.
                # If first observed inside, do NOT generate a false entry alert.
                self.track_states[track_id][zone.id] = is_inside
            elif not prev_inside and is_inside:
                # Actual transition: OUTSIDE -> INSIDE observed!
                alert = GeofenceAlert(
                    track_id=track_id,
                    label=label,
                    zone_id=zone.id,
                    zone_name=zone.name,
                    event_type="ZONE_ENTRY",
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    position=(int(center_point[0]), int(center_point[1])),
                )
                new_alerts.append(alert)
                self.alerts_history.append(alert)
                self.track_states[track_id][zone.id] = True
            else:
                # Either True -> True (staying inside), True -> False (leaving), or False -> False (outside)
                self.track_states[track_id][zone.id] = is_inside

        return new_alerts

    def cleanup_stale(
        self,
        stale_track_ids: Optional[Sequence[int]] = None,
        active_track_ids: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """
        Removes internal tracking state for stale or discontinued tracks.

        Can be called with either:
        - stale_track_ids: Explicit list of track IDs purged by TrackManager.
        - active_track_ids: List of currently valid track IDs (any other ID is purged).

        Returns:
            List of track IDs purged from the geofence state.
        """
        removed: List[int] = []

        if stale_track_ids is not None:
            for tid in stale_track_ids:
                if tid in self.track_states:
                    del self.track_states[tid]
                    removed.append(tid)
        elif active_track_ids is not None:
            active_set = set(active_track_ids)
            tracked = list(self.track_states.keys())
            for tid in tracked:
                if tid not in active_set:
                    del self.track_states[tid]
                    removed.append(tid)

        return removed

    def remove_track(self, track_id: int):
        """Remove geofence state for a specific track ID."""
        self.track_states.pop(track_id, None)

    def reset(self):
        """Clears all track state history and logged alerts for a fresh session."""
        self.track_states.clear()
        self.alerts_history.clear()

    def get_active_breaches(self) -> Dict[int, List[str]]:
        """
        Returns a mapping of active track IDs to the zone IDs they currently occupy.
        """
        breaches: Dict[int, List[str]] = {}
        for tid, z_map in self.track_states.items():
            inside_zones = [zid for zid, is_in in z_map.items() if is_in]
            if inside_zones:
                breaches[tid] = inside_zones
        return breaches

    def is_track_inside(self, track_id: int, zone_id: str) -> bool:
        """Returns True if the given track is currently inside the specified zone."""
        return self.track_states.get(track_id, {}).get(zone_id, False)

    def draw_geofences(
        self,
        frame: np.ndarray,
        show_labels: bool = True,
        fill_alpha: float = 0.15,
        active_breach_zones: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """
        Renders tactical geofence overlays on a video or camera frame.

        Args:
            frame: OpenCV BGR image array.
            show_labels: Whether to render tactical zone name headers.
            fill_alpha: Transparency factor for polygon interior shading.
            active_breach_zones: Zone IDs currently breached to highlight in alert red.

        Returns:
            The annotated frame.
        """
        if not self.zones:
            return frame

        breached_set = set(active_breach_zones or [])
        overlay = frame.copy()

        for zone in self.zones:
            if not zone.enabled:
                continue

            pts = np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
            is_breached = zone.id in breached_set

            # Use bright red if breached, otherwise zone color
            base_color = (20, 20, 230) if is_breached else zone.color_bgr
            line_thickness = 2 if not is_breached else 3

            # Fill translucent interior
            if fill_alpha > 0.0:
                cv2.fillPoly(overlay, [pts], base_color)

            # Draw tactical boundary
            cv2.polylines(frame, [pts], isClosed=True, color=base_color, thickness=line_thickness, lineType=cv2.LINE_AA)

            # Corner vertex indicators
            for pt in zone.polygon:
                vx, vy = int(pt[0]), int(pt[1])
                cv2.circle(frame, (vx, vy), 3, (240, 234, 216), -1, lineType=cv2.LINE_AA)

            # Draw tactical zone label badge
            if show_labels and len(zone.polygon) > 0:
                x_coords = [p[0] for p in zone.polygon]
                y_coords = [p[1] for p in zone.polygon]
                min_x = int(min(x_coords))
                min_y = int(min(y_coords))

                status_tag = "BREACHED" if is_breached else "RESTRICTED"
                tag_text = f"ZONE: {zone.name.upper()} [{status_tag}]"
                (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)

                badge_y1 = max(0, min_y - 18)
                badge_y2 = badge_y1 + 18
                badge_x2 = min_x + tw + 12

                bg_color = (20, 20, 90) if is_breached else (22, 36, 16)
                border_color = (0, 0, 220) if is_breached else base_color

                cv2.rectangle(frame, (min_x, badge_y1), (badge_x2, badge_y2), bg_color, -1)
                cv2.rectangle(frame, (min_x, badge_y1), (badge_x2, badge_y2), border_color, 1)
                text_color = (255, 255, 255) if is_breached else (240, 234, 216)
                cv2.putText(
                    frame,
                    tag_text,
                    (min_x + 6, badge_y1 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

        if fill_alpha > 0.0:
            cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)

        return frame
