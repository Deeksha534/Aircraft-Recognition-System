"""
movement.py
Movement Intelligence and Trajectory Analysis Module for AIRD.

IMPORTANT NOTE:
All coordinates, distances, vectors, headings, and velocities in this module
are measured strictly in 2D IMAGE-SPACE (pixel coordinates). They represent
screen-space target kinematics relative to the sensor frame and do NOT reflect
true 3D real-world geographic coordinates, radar track angles, or airspeed (km/h / knots).
"""

import math
from typing import Sequence, Tuple, Dict, Any, Union


def compute_motion_vector(
    trajectory: Sequence[Tuple[Union[int, float], Union[int, float]]],
    window: int = 4
) -> Tuple[float, float, float]:
    """
    Computes image-space displacement vector (dx, dy) and magnitude over a rolling window.

    Args:
        trajectory: Sequence of (cx, cy) 2D pixel center coordinates.
        window: Number of historical frame steps over which to measure movement.

    Returns:
        Tuple of (dx, dy, displacement_px) in pixels.
        Returns (0.0, 0.0, 0.0) if insufficient trajectory points exist.
    """
    if not trajectory or len(trajectory) < 2:
        return 0.0, 0.0, 0.0

    pts = list(trajectory)
    effective_window = min(max(1, int(window)), len(pts) - 1)

    p_curr = pts[-1]
    p_prev = pts[-1 - effective_window]

    dx = float(p_curr[0] - p_prev[0])
    dy = float(p_curr[1] - p_prev[1])
    displacement_px = float(math.hypot(dx, dy))

    return dx, dy, displacement_px


def get_cardinal_heading(
    dx: float,
    dy: float,
    stationary_threshold: float = 2.5
) -> str:
    """
    Determines 8-point cardinal/intercardinal image-space movement heading.

    Convention:
        - Image X-axis: increases to the right (East: +x, West: -x)
        - Image Y-axis: increases downward (North: -y, South: +y)

    Args:
        dx: Pixel displacement along X-axis.
        dy: Pixel displacement along Y-axis.
        stationary_threshold: Minimum displacement in pixels to be considered moving.

    Returns:
        One of: 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', or 'STATIONARY'.
    """
    disp = math.hypot(dx, dy)
    if disp < stationary_threshold:
        return "STATIONARY"

    # Invert dy to translate from computer vision (+y down) to standard cartesian (+y up)
    angle_deg = math.degrees(math.atan2(-dy, dx))
    # Normalize to [0, 360)
    angle_deg = (angle_deg + 360.0) % 360.0

    # 8 Cardinal Sectors (45 degrees each, centered on 0, 45, 90, etc.)
    if angle_deg >= 337.5 or angle_deg < 22.5:
        return "E"
    elif 22.5 <= angle_deg < 67.5:
        return "NE"
    elif 67.5 <= angle_deg < 112.5:
        return "N"
    elif 112.5 <= angle_deg < 157.5:
        return "NW"
    elif 157.5 <= angle_deg < 202.5:
        return "W"
    elif 202.5 <= angle_deg < 247.5:
        return "SW"
    elif 247.5 <= angle_deg < 292.5:
        return "S"
    elif 292.5 <= angle_deg < 337.5:
        return "SE"

    return "STATIONARY"


def compute_speed_px_sec(
    displacement_px: float,
    window: int,
    fps: Union[int, float]
) -> float:
    """
    Calculates image-space speed in pixels per second.

    Args:
        displacement_px: Euclidean displacement in pixels over the window.
        window: Frame window span over which displacement was measured.
        fps: Frames per second of the video/camera feed.

    Returns:
        Speed in pixels/second (px/s). Returns 0.0 for invalid/zero FPS or window.
    """
    if fps is None or fps <= 0 or window is None or window <= 0:
        return 0.0

    if displacement_px is None or displacement_px < 0:
        return 0.0

    effective_window = max(1, int(window))
    speed = (float(displacement_px) / float(effective_window)) * float(fps)
    return float(speed)


def compute_total_distance(
    trajectory: Sequence[Tuple[Union[int, float], Union[int, float]]]
) -> float:
    """
    Calculates cumulative image-space path length across all trajectory points.

    Args:
        trajectory: Sequence of (cx, cy) pixel points.

    Returns:
        Cumulative Euclidean distance in pixels.
    """
    if not trajectory or len(trajectory) < 2:
        return 0.0

    pts = list(trajectory)
    total_dist = 0.0
    for i in range(1, len(pts)):
        p1 = pts[i - 1]
        p2 = pts[i]
        total_dist += math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    return float(total_dist)


def get_movement_summary(
    trajectory: Sequence[Tuple[Union[int, float], Union[int, float]]],
    fps: Union[int, float] = 25.0,
    window: int = 4,
    stationary_threshold: float = 2.5
) -> Dict[str, Any]:
    """
    Computes a comprehensive movement intelligence summary dictionary for a track.

    Args:
        trajectory: Sequence of (cx, cy) pixel points.
        fps: Video/camera feed frames per second (default: 25.0).
        window: Historical point span for motion vector calculation.
        stationary_threshold: Threshold below which target is marked STATIONARY.

    Returns:
        Dictionary with:
            - dx: float
            - dy: float
            - displacement_px: float
            - heading: str
            - speed_px_sec: float
            - total_distance_px: float
            - status: 'INITIALIZING' | 'MOVING' | 'STATIONARY'
    """
    if not trajectory or len(trajectory) < 2:
        return {
            "dx": 0.0,
            "dy": 0.0,
            "displacement_px": 0.0,
            "heading": "STATIONARY",
            "speed_px_sec": 0.0,
            "total_distance_px": 0.0,
            "status": "INITIALIZING"
        }

    pts = list(trajectory)
    effective_window = min(max(1, int(window)), len(pts) - 1)

    dx, dy, disp = compute_motion_vector(pts, window=effective_window)
    heading = get_cardinal_heading(dx, dy, stationary_threshold=stationary_threshold)
    speed = compute_speed_px_sec(disp, effective_window, fps)
    total_dist = compute_total_distance(pts)

    status = "STATIONARY" if heading == "STATIONARY" else "MOVING"

    return {
        "dx": round(dx, 2),
        "dy": round(dy, 2),
        "displacement_px": round(disp, 2),
        "heading": heading,
        "speed_px_sec": round(speed, 2),
        "total_distance_px": round(total_dist, 2),
        "status": status
    }
