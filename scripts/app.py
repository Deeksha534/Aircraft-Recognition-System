import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import sys
import time
import json

# Ensure scripts directory is in sys.path for local module resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tracking import TrackManager, TrackState
from movement import get_movement_summary
from geofencing import GeofenceManager, GeofenceZone, GeofenceAlert

# ─────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DETECTOR_PATH  = os.path.join(PROJECT_ROOT, "runs", "detect", "AircraftDetector3", "weights", "best.pt")
if not os.path.exists(DETECTOR_PATH):
    DETECTOR_PATH = os.path.join(PROJECT_ROOT, "models", "detector.pt")

CLASSIFIER_PATH = os.path.join(PROJECT_ROOT, "models", "aircraft_classifier.h5")
CLASSES_PATH    = os.path.join(PROJECT_ROOT, "config", "classes.json")
SPECS_PATH      = os.path.join(PROJECT_ROOT, "config", "aircraft_specs.json")
GEOFENCES_PATH  = os.path.join(PROJECT_ROOT, "config", "geofences.json")
OUTPUT_DIR      = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_classes():
    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(f"Configuration file not found: {CLASSES_PATH}")
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)
    return classes


CLASS_NAMES = load_classes()


def load_aircraft_specs():
    """Safely loads structured aircraft specifications database with fallback protection."""
    if not os.path.exists(SPECS_PATH):
        return {
            "Unknown": {
                "display_name": "Unidentified Aerial Target",
                "manufacturer": "Unknown / Unclassified",
                "origin": "Unidentified",
                "category": "Unclassified Target",
                "role": "Unidentified Aerial System / Contact",
                "crew": "Unknown",
                "wingspan": "Unknown",
                "length": "Unknown",
                "max_speed": "Unknown",
                "service_ceiling": "Unknown",
                "range": "Unknown",
                "powerplant": "Unknown",
                "primary_armament": "Unknown",
                "description": "Target specification profile not available."
            }
        }
    try:
        with open(SPECS_PATH, "r", encoding="utf-8") as f:
            specs = json.load(f)
        return specs
    except Exception as e:
        print(f"Warning: Failed to load {SPECS_PATH}: {e}")
        return {}


AIRCRAFT_SPECS = load_aircraft_specs()


def get_aircraft_specs(label: str) -> dict:
    """Safe lookup helper to retrieve specification dictionary for a given aircraft class label."""
    fallback = AIRCRAFT_SPECS.get("Unknown", {
        "display_name": "Unidentified Aerial Target",
        "manufacturer": "Unknown / Unclassified",
        "origin": "Unidentified",
        "category": "Unclassified Target",
        "role": "Unidentified Aerial System / Contact",
        "crew": "Unknown",
        "wingspan": "Unknown",
        "length": "Unknown",
        "max_speed": "Unknown",
        "service_ceiling": "Unknown",
        "range": "Unknown",
        "powerplant": "Unknown",
        "primary_armament": "Unknown",
        "description": "Target specification profile not available."
    })
    return AIRCRAFT_SPECS.get(label, fallback)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AIRD · Aircraft Intelligence System",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# MILITARY OPS THEME CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;500;600;700&display=swap');

:root {
    --olive:        #4a5c2f;
    --olive-light:  #6b7c45;
    --olive-dark:   #2e3a1c;
    --amber:        #d4840a;
    --amber-bright: #f5a623;
    --green-tac:    #7ab648;
    --red-alert:    #c0392b;
    --bg-base:      #1a1e14;
    --bg-panel:     #222818;
    --bg-card:      #1e2416;
    --border:       rgba(180,160,80,.2);
    --border-hi:    rgba(212,132,10,.55);
    --txt:          #f0ead8;
    --txt-sec:      #b8aa88;
    --txt-dim:      rgba(240,234,216,.38);
    --txt-label:    #8a9060;
}

/* ── BODY ── */
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: var(--bg-base) !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    color: var(--txt) !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        repeating-linear-gradient(0deg,transparent 0px,transparent 3px,rgba(0,0,0,.06) 3px,rgba(0,0,0,.06) 4px),
        repeating-linear-gradient(90deg,transparent 0px,transparent 60px,rgba(255,255,255,.012) 60px,rgba(255,255,255,.012) 61px);
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #161a10 !important;
    border-right: 2px solid #2e3a1c !important;
}
[data-testid="stSidebar"] * { color: var(--txt) !important; }
[data-testid="stSidebar"] .stRadio label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 2px;
}

/* ── TEXT ── */
h1,h2,h3,h4 {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 3px !important;
    color: var(--txt) !important;
}
p, span, div, label {
    font-family: 'Barlow Condensed', sans-serif !important;
    color: var(--txt) !important;
}

/* ── BUTTONS ── */
.stButton > button {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 15px !important;
    letter-spacing: 3px !important;
    color: #1a1e14 !important;
    background: linear-gradient(135deg, #d4840a, #f5a623) !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 10px 28px !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #f5a623, #ffc84a) !important;
    box-shadow: 0 0 20px rgba(212,132,10,.4) !important;
    transform: translateY(-1px);
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-card) !important;
    border: 2px dashed rgba(180,160,80,.35) !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--amber) !important;
    background: rgba(212,132,10,.05) !important;
}
[data-testid="stFileUploaderDropzone"] [data-testid="stIconMaterial"] {
    display: none !important;
}
[data-testid="stFileUploaderDropzone"] > div > span {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
    color: var(--txt-dim) !important;
    letter-spacing: 2px !important;
}

/* ── PROGRESS ── */
[data-testid="stProgressBar"] > div {
    background: rgba(180,160,80,.12) !important;
    border-radius: 2px !important;
}
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #4a5c2f, #f5a623) !important;
}

/* ── SLIDERS ── */
[data-testid="stSlider"] > div > div > div { background: var(--amber) !important; }
[data-testid="stSlider"] [role="slider"] {
    background: #f5a623 !important;
    border-color: #d4840a !important;
    box-shadow: 0 0 8px rgba(212,132,10,.5) !important;
}

/* ── ALERTS ── */
[data-testid="stAlert"] {
    background: rgba(212,132,10,.08) !important;
    border: 1px solid rgba(212,132,10,.3) !important;
    border-left: 3px solid #d4840a !important;
    border-radius: 4px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid rgba(180,160,80,.2) !important;
    border-top: 2px solid #6b7c45 !important;
    border-radius: 4px !important;
    padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] > div {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 3px !important;
    color: #8a9060 !important;
}
[data-testid="stMetricValue"] > div {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 28px !important;
    letter-spacing: 2px !important;
    color: #f5a623 !important;
}

/* ── DOWNLOAD BUTTON ── */
[data-testid="stDownloadButton"] > button {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 14px !important;
    letter-spacing: 3px !important;
    background: transparent !important;
    border: 1px solid rgba(212,132,10,.55) !important;
    color: #f5a623 !important;
    border-radius: 4px !important;
    padding: 9px 22px !important;
    transition: all .2s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(212,132,10,.1) !important;
}

/* ── IMAGES / VIDEO ── */
[data-testid="stImage"] img {
    border-radius: 4px !important;
    border: 1px solid rgba(180,160,80,.2) !important;
}
video {
    border-radius: 4px !important;
    border: 1px solid rgba(180,160,80,.2) !important;
}

/* ── CHECKBOX ── */
.stCheckbox label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    color: #b8aa88 !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: #4a5c2f; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #6b7c45; }

hr { border-color: rgba(180,160,80,.15) !important; }

@keyframes blink {0%,100%{opacity:1}50%{opacity:.25}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HTML HELPER COMPONENTS
# ─────────────────────────────────────────────

def render_header():
    now  = time.strftime("%H:%M:%S UTC")
    date = time.strftime("%d %b %Y").upper()
    st.markdown(f"""
    <div style="background:#161a10; border-bottom:2px solid #2e3a1c;
                padding:14px 24px; margin:-1rem -1rem 28px -1rem;
                display:flex; align-items:center; justify-content:space-between;">

      <div style="display:flex; align-items:center; gap:16px;">
        <div style="width:48px;height:48px;background:#2e3a1c;
                    border:2px solid #4a5c2f;border-radius:4px;
                    display:flex;align-items:center;justify-content:center;">
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none"
               stroke="#d4840a" stroke-width="2">
            <path d="M21 16v-2l-8-5V3.5a1.5 1.5 0 0 0-3 0V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5z"/>
          </svg>
        </div>
        <div>
          <div style="font-family:'Bebas Neue',sans-serif;font-size:26px;
                      letter-spacing:5px;color:#f0ead8;line-height:1;">
            AIRCRAFT INTELLIGENCE SYSTEM
          </div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:9px;
                      letter-spacing:4px;color:#6b7c45;margin-top:2px;">
            TACTICAL AI RECOGNITION & TRACKING DASHBOARD · v2.0
          </div>
        </div>
      </div>

      <div style="display:flex;align-items:center;gap:28px;
                  font-family:'Share Tech Mono',monospace;">
        <div style="text-align:right;">
          <div style="font-size:9px;letter-spacing:3px;color:#6b7c45;">MISSION TIME</div>
          <div style="font-size:16px;letter-spacing:3px;color:#f5a623;margin-top:2px;">{now}</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:9px;letter-spacing:3px;color:#6b7c45;">DATE</div>
          <div style="font-size:13px;letter-spacing:2px;color:#b8aa88;margin-top:2px;">{date}</div>
        </div>
        <div style="display:flex;flex-direction:column;gap:6px;">
          <div style="display:flex;align-items:center;gap:7px;font-size:9px;
                      letter-spacing:2px;color:#7ab648;">
            <div style="width:6px;height:6px;border-radius:50%;background:#7ab648;
                        box-shadow:0 0 6px #7ab648;animation:blink 1.8s infinite;"></div>
            SYS ONLINE
          </div>
          <div style="display:flex;align-items:center;gap:7px;font-size:9px;
                      letter-spacing:2px;color:#f5a623;">
            <div style="width:6px;height:6px;border-radius:50%;background:#f5a623;
                        box-shadow:0 0 6px #f5a623;animation:blink 2.4s infinite;"></div>
            TRACKER ACTIVE
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def section_divider(title, icon="▶"):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:20px 0 16px;">
      <span style="font-family:'Bebas Neue',sans-serif;font-size:11px;
                   letter-spacing:2px;color:#d4840a;">{icon}</span>
      <span style="font-family:'Bebas Neue',sans-serif;font-size:16px;
                   letter-spacing:4px;color:#b8aa88;">{title}</span>
      <div style="flex:1;height:1px;
                  background:linear-gradient(90deg,rgba(180,160,80,.3),transparent);"></div>
    </div>
    """, unsafe_allow_html=True)


def target_card(label, conf, index, bbox, track_id=None, det_conf=None, specs=None, movement_info=None, breaching_zones=None):
    if specs is None:
        specs = get_aircraft_specs(label)
    display_title = specs.get("display_name", label)
    origin = specs.get("origin", "N/A")
    role = specs.get("role", "N/A")
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    conf_bar  = int(conf * 100)
    conf_color = "#c0392b" if breaching_zones else ("#7ab648" if conf > 0.7 else "#f5a623" if conf > 0.45 else "#c0392b")
    tag_label = f"TRACK #{track_id:02d}" if track_id is not None else f"TGT {index:02d}"
    det_html = f"<span>DET &nbsp;<span style='color:#b8aa88;'>{det_conf:.0%}</span></span>" if det_conf is not None else ""

    # Optional compact movement intelligence row (for Video / Live Tracking)
    movement_html = ""
    if movement_info is not None:
        status = movement_info.get("status", "INITIALIZING")
        status_color = "#f5a623" if status == "MOVING" else "#7ab648" if status == "STATIONARY" else "#8a9060"
        heading_text = movement_info.get("heading", "—") if status == "MOVING" else "—"
        speed_text = f"{movement_info.get('speed_px_sec', 0.0):.1f} px/s" if status != "INITIALIZING" else "—"
        movement_html = f"""
        <div style="margin-top:8px;padding-top:6px;border-top:1px dashed rgba(180,160,80,.2);
                    display:flex;justify-content:space-between;align-items:center;
                    font-size:9px;letter-spacing:1px;color:#8a9060;">
          <span>STATUS &nbsp;<span style="color:{status_color};font-weight:600;">{status}</span></span>
          <span>HDG &nbsp;<span style="color:#b8aa88;font-weight:600;">{heading_text}</span></span>
          <span>SPEED &nbsp;<span style="color:#7ab648;font-weight:600;">{speed_text}</span></span>
        </div>
        """

    # Optional restricted zone breach alert badge
    breach_html = ""
    if breaching_zones:
        breach_names = ", ".join(breaching_zones)
        breach_html = f"""
        <div style="margin-top:8px;padding:4px 8px;background:rgba(192,57,43,.25);
                    border:1px solid #c0392b;border-radius:3px;font-size:9px;color:#f0ead8;">
          <span style="color:#e07060;font-weight:700;">⚠ RESTRICTED ZONE:</span> {breach_names}
        </div>
        """

    st.markdown(f"""
    <div style="background:#1e2416;border:1px solid rgba(180,160,80,.2);
                border-left:3px solid {conf_color};border-radius:4px;
                padding:12px 14px;margin-bottom:8px;
                font-family:'Share Tech Mono',monospace;">
      <div style="display:flex;justify-content:space-between;
                  align-items:center;margin-bottom:4px;">
        <span style="font-family:'Bebas Neue',sans-serif;font-size:17px;
                     letter-spacing:2px;color:#f0ead8;">{display_title}</span>
        <span style="font-size:9px;letter-spacing:2px;color:#6b7c45;
                     background:#2e3a1c;padding:3px 8px;border-radius:3px;">
          {tag_label}
        </span>
      </div>
      <div style="font-size:9px;letter-spacing:1px;color:#8a9060;margin-bottom:8px;">
        {role} &nbsp;·&nbsp; <span style="color:#b8aa88;">{origin}</span>
      </div>
      <div style="margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
          <span style="font-size:9px;letter-spacing:2px;color:#6b7c45;">CONFIDENCE</span>
          <span style="font-size:10px;color:{conf_color};">{conf:.1%}</span>
        </div>
        <div style="height:3px;background:rgba(255,255,255,.08);
                    border-radius:1px;overflow:hidden;">
          <div style="width:{conf_bar}%;height:100%;background:{conf_color};
                      border-radius:1px;"></div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;
                  font-size:9px;letter-spacing:1px;color:#8a9060;">
        <span>POS &nbsp;<span style="color:#b8aa88;">{x1},{y1}</span></span>
        <span>DIM &nbsp;<span style="color:#b8aa88;">{w}x{h}px</span></span>
        {det_html}
      </div>
      {movement_html}
      {breach_html}
    </div>
    """, unsafe_allow_html=True)


def render_fleet_movement_summary(tracked_summary: dict):
    """Renders a compact, tactical Fleet Movement Intelligence summary table."""
    if not tracked_summary:
        return

    section_divider("MOVEMENT INTELLIGENCE — FLEET KINEMATICS", icon="⬡")

    rows_html = ""
    for tid, data in tracked_summary.items():
        specs = data.get("specs") or get_aircraft_specs(data.get("label", "Unknown"))
        display_name = specs.get("display_name", data.get("label", "Unknown"))
        conf = data.get("conf", 0.0)
        move = data.get("move_info") or {}
        status = move.get("status", "INITIALIZING")
        heading = move.get("heading", "—") if status == "MOVING" else "—"
        speed = f"{move.get('speed_px_sec', 0.0):.1f} px/s" if status != "INITIALIZING" else "—"
        dist = f"{move.get('total_distance_px', 0.0):.0f} px"

        status_color = "#f5a623" if status == "MOVING" else "#7ab648" if status == "STATIONARY" else "#8a9060"

        rows_html += f"""
        <tr style="border-bottom:1px solid rgba(180,160,80,.15);font-family:'Share Tech Mono',monospace;font-size:11px;">
          <td style="padding:8px 10px;color:#d4840a;font-weight:600;">TRACK #{tid:02d}</td>
          <td style="padding:8px 10px;color:#f0ead8;font-family:'Barlow Condensed',sans-serif;font-size:14px;font-weight:600;">{display_name}</td>
          <td style="padding:8px 10px;color:#b8aa88;">{conf:.0%}</td>
          <td style="padding:8px 10px;color:#f5a623;font-weight:600;">{heading}</td>
          <td style="padding:8px 10px;color:#7ab648;">{speed}</td>
          <td style="padding:8px 10px;color:#b8aa88;">{dist}</td>
          <td style="padding:8px 10px;">
            <span style="background:rgba(0,0,0,.3);border:1px solid {status_color};color:{status_color};padding:2px 8px;border-radius:3px;font-size:9px;letter-spacing:1px;">
              {status}
            </span>
          </td>
        </tr>
        """

    st.markdown(f"""
    <div style="background:#1e2416;border:1px solid rgba(180,160,80,.25);border-radius:4px;padding:14px 16px;margin-bottom:18px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
        <span style="font-family:'Share Tech Mono',monospace;font-size:9px;letter-spacing:2px;color:#6b7c45;">
          SENSOR FRAME IMAGE-SPACE KINEMATICS
        </span>
        <span style="font-family:'Share Tech Mono',monospace;font-size:9px;letter-spacing:2px;color:#8a9060;">
          TOTAL TRACKS: {len(tracked_summary):02d}
        </span>
      </div>
      <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;text-align:left;">
          <thead>
            <tr style="border-bottom:2px solid #2e3a1c;font-family:'Share Tech Mono',monospace;font-size:9px;letter-spacing:2px;color:#8a9060;">
              <th style="padding:6px 10px;">TRACK ID</th>
              <th style="padding:6px 10px;">CLASSIFICATION</th>
              <th style="padding:6px 10px;">CONF</th>
              <th style="padding:6px 10px;">HEADING</th>
              <th style="padding:6px 10px;">IMAGE SPEED</th>
              <th style="padding:6px 10px;">TOTAL PATH</th>
              <th style="padding:6px 10px;">STATUS</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_geofence_alerts_summary(alerts: list):
    """Renders tactical Restricted-Zone Entry Alerts incident log table."""
    section_divider("RESTRICTED AIRSPACE PERIMETER BREACH INCIDENTS", icon="⚠")

    if not alerts:
        st.markdown("""
        <div style="background:rgba(122,182,72,.08);border:1px solid rgba(122,182,72,.3);
                    border-left:3px solid #7ab648;border-radius:4px;padding:12px 16px;
                    margin-bottom:18px;font-family:'Share Tech Mono',monospace;">
          <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:8px;height:8px;border-radius:50%;background:#7ab648;box-shadow:0 0 6px #7ab648;"></div>
            <span style="font-size:11px;letter-spacing:2px;color:#7ab648;font-weight:600;">
              ALL PERIMETERS SECURE — 0 RESTRICTED-ZONE ENTRY VIOLATIONS DETECTED
            </span>
          </div>
          <div style="font-size:9px;color:#8a9060;margin-top:4px;letter-spacing:1px;">
            Target kinematic paths did not breach any active restricted geofence sectors.
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown(f"""
    <div style="background:rgba(192,57,43,.12);border:1px solid rgba(192,57,43,.4);
                border-left:3px solid #c0392b;border-radius:4px;padding:12px 16px;
                margin-bottom:14px;font-family:'Share Tech Mono',monospace;">
      <div style="display:flex;align-items:center;gap:10px;">
        <div style="width:8px;height:8px;border-radius:50%;background:#c0392b;box-shadow:0 0 8px #c0392b;animation:blink 1.5s infinite;"></div>
        <span style="font-size:12px;letter-spacing:2px;color:#f0ead8;font-weight:700;">
          SECURITY ALERT — {len(alerts)} RESTRICTED-ZONE ENTRY INCIDENT(S) LOGGED
        </span>
      </div>
      <div style="font-size:9px;color:#e07060;margin-top:4px;letter-spacing:1px;">
        Target outside-to-inside transitions across restricted perimeters recorded.
      </div>
    </div>
    """, unsafe_allow_html=True)

    rows_html = ""
    for idx, alert in enumerate(alerts, 1):
        pos_str = f"({alert.position[0]}, {alert.position[1]}) px" if hasattr(alert, "position") else "—"
        rows_html += f"""
        <tr style="border-bottom:1px solid rgba(180,160,80,.15);font-family:'Share Tech Mono',monospace;font-size:11px;">
          <td style="padding:8px 10px;color:#b8aa88;">#{idx:02d}</td>
          <td style="padding:8px 10px;color:#d4840a;font-weight:600;">TRACK #{alert.track_id:02d}</td>
          <td style="padding:8px 10px;color:#f0ead8;font-family:'Barlow Condensed',sans-serif;font-size:14px;font-weight:600;">{alert.label}</td>
          <td style="padding:8px 10px;color:#e07060;font-weight:600;">{alert.zone_name}</td>
          <td style="padding:8px 10px;">
            <span style="background:rgba(192,57,43,.25);border:1px solid #c0392b;color:#f5a623;padding:2px 8px;border-radius:3px;font-size:9px;letter-spacing:1px;font-weight:700;">
              {alert.event_type}
            </span>
          </td>
          <td style="padding:8px 10px;color:#b8aa88;">F:{alert.frame_idx:04d}</td>
          <td style="padding:8px 10px;color:#7ab648;">{alert.timestamp}</td>
          <td style="padding:8px 10px;color:#8a9060;">{pos_str}</td>
        </tr>
        """

    st.markdown(f"""
    <div style="background:#1e2416;border:1px solid rgba(180,160,80,.25);border-radius:4px;padding:14px 16px;margin-bottom:18px;">
      <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;text-align:left;">
          <thead>
            <tr style="border-bottom:2px solid #2e3a1c;font-family:'Share Tech Mono',monospace;font-size:9px;letter-spacing:2px;color:#8a9060;">
              <th style="padding:6px 10px;">INCIDENT</th>
              <th style="padding:6px 10px;">TRACK ID</th>
              <th style="padding:6px 10px;">CLASSIFICATION</th>
              <th style="padding:6px 10px;">RESTRICTED ZONE</th>
              <th style="padding:6px 10px;">EVENT</th>
              <th style="padding:6px 10px;">FRAME</th>
              <th style="padding:6px 10px;">TIME</th>
              <th style="padding:6px 10px;">COORDINATES</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_intelligence_dossier(specs: dict, title_prefix: str = ""):
    """Renders a comprehensive, tactically styled intelligence dossier for an aircraft."""
    name = specs.get("display_name", "Unidentified Target")
    mfr = specs.get("manufacturer", "N/A")
    origin = specs.get("origin", "N/A")
    cat = specs.get("category", "N/A")
    role = specs.get("role", "N/A")
    crew = specs.get("crew", "N/A")
    span = specs.get("wingspan", "N/A")
    length = specs.get("length", "N/A")
    speed = specs.get("max_speed", "N/A")
    ceiling = specs.get("service_ceiling", "N/A")
    range_val = specs.get("range", "N/A")
    engine = specs.get("powerplant", "N/A")
    weapons = specs.get("primary_armament", "N/A")
    desc = specs.get("description", "No intelligence report available.")

    header_text = f"{title_prefix} · {name.upper()}" if title_prefix else name.upper()

    st.markdown(f"""
    <div style="background:#1e2416;border:1px solid rgba(180,160,80,.3);
                border-top:3px solid #d4840a;border-radius:4px;
                padding:16px 18px;margin-top:12px;margin-bottom:16px;
                font-family:'Share Tech Mono',monospace;">
      
      <div style="display:flex;justify-content:space-between;align-items:center;
                  border-bottom:1px solid rgba(180,160,80,.2);padding-bottom:10px;margin-bottom:14px;">
        <div>
          <div style="font-size:9px;letter-spacing:3px;color:#d4840a;">TACTICAL INTELLIGENCE DOSSIER</div>
          <div style="font-family:'Bebas Neue',sans-serif;font-size:22px;letter-spacing:2px;color:#f0ead8;">
            {header_text}
          </div>
        </div>
        <div style="text-align:right;">
          <span style="font-size:9px;letter-spacing:2px;color:#7ab648;background:#2e3a1c;
                       padding:4px 10px;border-radius:3px;border:1px solid rgba(122,182,72,.3);">
            INTEL PROFILE
          </span>
        </div>
      </div>

      <!-- Quick Specs Grid -->
      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));gap:8px;margin-bottom:12px;">
        <div style="background:#161a10;border:1px solid rgba(180,160,80,.15);border-radius:3px;padding:8px 10px;">
          <div style="font-size:8px;letter-spacing:2px;color:#6b7c45;">MANUFACTURER</div>
          <div style="font-size:11px;color:#f0ead8;margin-top:2px;">{mfr}</div>
        </div>
        <div style="background:#161a10;border:1px solid rgba(180,160,80,.15);border-radius:3px;padding:8px 10px;">
          <div style="font-size:8px;letter-spacing:2px;color:#6b7c45;">ORIGIN / NATION</div>
          <div style="font-size:11px;color:#f0ead8;margin-top:2px;">{origin}</div>
        </div>
        <div style="background:#161a10;border:1px solid rgba(180,160,80,.15);border-radius:3px;padding:8px 10px;">
          <div style="font-size:8px;letter-spacing:2px;color:#6b7c45;">CATEGORY</div>
          <div style="font-size:11px;color:#f5a623;margin-top:2px;">{cat}</div>
        </div>
        <div style="background:#161a10;border:1px solid rgba(180,160,80,.15);border-radius:3px;padding:8px 10px;">
          <div style="font-size:8px;letter-spacing:2px;color:#6b7c45;">PRIMARY ROLE</div>
          <div style="font-size:11px;color:#f0ead8;margin-top:2px;">{role}</div>
        </div>
        <div style="background:#161a10;border:1px solid rgba(180,160,80,.15);border-radius:3px;padding:8px 10px;">
          <div style="font-size:8px;letter-spacing:2px;color:#6b7c45;">CREW COMPLEMENT</div>
          <div style="font-size:11px;color:#b8aa88;margin-top:2px;">{crew}</div>
        </div>
        <div style="background:#161a10;border:1px solid rgba(180,160,80,.15);border-radius:3px;padding:8px 10px;">
          <div style="font-size:8px;letter-spacing:2px;color:#6b7c45;">MAX SPEED</div>
          <div style="font-size:11px;color:#7ab648;margin-top:2px;">{speed}</div>
        </div>
        <div style="background:#161a10;border:1px solid rgba(180,160,80,.15);border-radius:3px;padding:8px 10px;">
          <div style="font-size:8px;letter-spacing:2px;color:#6b7c45;">SERVICE CEILING</div>
          <div style="font-size:11px;color:#b8aa88;margin-top:2px;">{ceiling}</div>
        </div>
        <div style="background:#161a10;border:1px solid rgba(180,160,80,.15);border-radius:3px;padding:8px 10px;">
          <div style="font-size:8px;letter-spacing:2px;color:#6b7c45;">OPERATIONAL RANGE</div>
          <div style="font-size:11px;color:#b8aa88;margin-top:2px;">{range_val}</div>
        </div>
        <div style="background:#161a10;border:1px solid rgba(180,160,80,.15);border-radius:3px;padding:8px 10px;">
          <div style="font-size:8px;letter-spacing:2px;color:#6b7c45;">DIMENSIONS</div>
          <div style="font-size:11px;color:#b8aa88;margin-top:2px;">Span: {span} · Len: {length}</div>
        </div>
      </div>

      <!-- Powerplant & Armament Details -->
      <div style="display:grid;grid-template-columns:1fr;gap:8px;margin-bottom:12px;">
        <div style="background:#161a10;border:1px solid rgba(180,160,80,.15);border-radius:3px;padding:8px 10px;">
          <span style="font-size:8px;letter-spacing:2px;color:#6b7c45;display:block;margin-bottom:2px;">POWERPLANT</span>
          <span style="font-size:11px;color:#d4840a;">{engine}</span>
        </div>
        <div style="background:#161a10;border:1px solid rgba(180,160,80,.15);border-radius:3px;padding:8px 10px;">
          <span style="font-size:8px;letter-spacing:2px;color:#6b7c45;display:block;margin-bottom:2px;">ARMAMENT / PAYLOAD SUITE</span>
          <span style="font-size:11px;color:#f0ead8;">{weapons}</span>
        </div>
      </div>

      <!-- Tactical Description -->
      <div style="background:#161a10;border-left:2px solid #7ab648;border-radius:0 3px 3px 0;padding:10px 12px;">
        <div style="font-size:8px;letter-spacing:2px;color:#6b7c45;margin-bottom:4px;">OPERATIONAL OVERVIEW</div>
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:13px;color:#b8aa88;line-height:1.4;">
          {desc}
        </div>
      </div>

    </div>
    """, unsafe_allow_html=True)


def empty_state(message):
    st.markdown(f"""
    <div style="border:1px dashed rgba(180,160,80,.2);border-radius:6px;
                padding:56px 20px;text-align:center;background:rgba(0,0,0,.15);">
      <div style="font-size:40px;opacity:.15;margin-bottom:14px;">✈</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:10px;
                  letter-spacing:3px;color:rgba(240,234,216,.25);">{message}</div>
    </div>
    """, unsafe_allow_html=True)


def feed_header(label="DETECTION FEED ACTIVE"):
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                font-family:'Share Tech Mono',monospace;font-size:9px;
                letter-spacing:2px;color:#6b7c45;
                border-top:1px solid rgba(180,160,80,.15);
                border-bottom:1px solid rgba(180,160,80,.15);
                padding:6px 0;margin-bottom:10px;">
      <span>◉ {label}</span>
      <span>CONF THRESHOLD ACTIVE · YOLOv8 + BYTETRACK + CNN</span>
      <span>RGB OUTPUT</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    detector   = YOLO(DETECTOR_PATH)
    classifier = load_model(CLASSIFIER_PATH)

    # Validate class count matches output dimensions
    expected_classes = classifier.output_shape[-1]
    if len(CLASS_NAMES) != expected_classes:
        raise ValueError(
            f"Class count mismatch: config/classes.json defines {len(CLASS_NAMES)} classes, "
            f"but classifier model output shape expects {expected_classes} classes."
        )
    return detector, classifier


def classify_crop(crop, classifier):
    resized = cv2.resize(crop, (224, 224))
    arr     = image.img_to_array(resized) / 255.0
    arr     = np.expand_dims(arr, axis=0)
    preds   = classifier.predict(arr, verbose=False)[0]
    idx     = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx])


# ─────────────────────────────────────────────
# BOOT
# ─────────────────────────────────────────────
render_header()

with st.spinner("LOADING INFERENCE PIPELINE..."):
    detector, classifier = load_models()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 4px;">
      <div style="font-family:'Bebas Neue',sans-serif;font-size:22px;
                  letter-spacing:5px;color:#f0ead8;padding:0 4px;">NAVIGATION</div>
      <div style="height:2px;background:linear-gradient(90deg,#4a5c2f,transparent);
                  margin-top:6px;margin-bottom:20px;"></div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio("MODE", ["IMAGE", "VIDEO", "LIVE CAMERA"], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:9px;
                letter-spacing:3px;color:#6b7c45;margin-bottom:12px;">
                INFERENCE PARAMS</div>""", unsafe_allow_html=True)

    conf_thresh = st.slider("Confidence Threshold", 0.10, 0.90, 0.25, 0.05)
    iou_thresh  = st.slider("IoU Threshold",         0.10, 0.90, 0.45, 0.05)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:9px;
                letter-spacing:3px;color:#6b7c45;margin-bottom:14px;">
                SYSTEM MANIFEST</div>""", unsafe_allow_html=True)

    for row_label, val, pct, color in [
        ("DETECTOR",   "YOLOv8 + ByteTrack", 95, "#7ab648"),
        ("CLASSIFIER", "CNN / KERAS",        85, "#7ab648"),
        ("CLASSES",    str(len(CLASS_NAMES)),min(len(CLASS_NAMES)*9, 100), "#f5a623"),
        ("GEOFENCING", "Active (Perimeters)", 100, "#7ab648"),
        ("BACKEND",    "TensorFlow",         95, "#7ab648"),
    ]:
        st.markdown(f"""
        <div style="margin-bottom:13px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
            <span style="font-family:'Share Tech Mono',monospace;font-size:9px;
                         letter-spacing:2px;color:#6b7c45;">{row_label}</span>
            <span style="font-family:'Share Tech Mono',monospace;font-size:10px;
                         color:#b8aa88;">{val}</span>
          </div>
          <div style="height:2px;background:rgba(255,255,255,.06);border-radius:1px;overflow:hidden;">
            <div style="width:{pct}%;height:100%;background:{color};"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:9px;
                letter-spacing:3px;color:#6b7c45;margin-bottom:12px;">
                SESSION LOG</div>""", unsafe_allow_html=True)

    if "detections_total" not in st.session_state:
        st.session_state.detections_total = 0
        st.session_state.frames_processed = 0

    col_a, col_b = st.columns(2)
    col_a.metric("TARGETS",  st.session_state.detections_total)
    col_b.metric("FRAMES",   st.session_state.frames_processed)


# ─────────────────────────────────────────────
# MODE: IMAGE
# ─────────────────────────────────────────────
if mode == "IMAGE":

    section_divider("SENSOR INPUT — IMAGE FEED")

    uploaded = st.file_uploader(
        "UPLOAD TARGET IMAGE", type=["jpg","jpeg","png"],
        label_visibility="collapsed",
    )

    if not uploaded:
        empty_state("DROP A JPG · PNG IMAGE TO BEGIN TARGET ANALYSIS")

    else:
        col_main, col_info = st.columns([2, 1], gap="large")

        with col_main:
            section_divider("ANNOTATED OUTPUT")
            feed_header("STATIC TARGET SCAN")

            with st.spinner("RUNNING DETECTION & RECOGNITION..."):
                img     = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)
                results = detector(img, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]

                detected = []
                for box in results.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    label, conf = classify_crop(crop, classifier)
                    specs = get_aircraft_specs(label)
                    detected.append((label, conf, (x1, y1, x2, y2), specs))

                    # Bounding box — tactical amber
                    cv2.rectangle(img, (x1,y1), (x2,y2), (10,132,212), 2)
                    # Corner brackets
                    L = 16
                    for (px,py,dx,dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                        cv2.line(img,(px,py),(px+dx*L,py),(10,132,212),2)
                        cv2.line(img,(px,py),(px,py+dy*L),(10,132,212),2)
                    # Label banner with high-contrast tactical styling and boundary safety
                    H_img, W_img = img.shape[:2]
                    txt = f"{label} | {conf:.0%}"
                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
                    bw, bh = tw + 16, th + 12
                    by1 = y1 - bh if y1 - bh >= 0 else y1
                    by2 = by1 + bh
                    bx1 = max(0, min(x1, W_img - bw))
                    bx2 = min(W_img, bx1 + bw)
                    ty = by2 - 6 if y1 - bh >= 0 else by1 + th + 6
                    cv2.rectangle(img, (bx1, by1), (bx2, by2), (46, 76, 28), -1)
                    cv2.rectangle(img, (bx1, by1), (bx2, by2), (10, 132, 212), 1)
                    cv2.putText(img, txt, (bx1 + 8, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (240, 234, 216), 2, cv2.LINE_AA)

                st.session_state.detections_total += len(detected)
                st.session_state.frames_processed += 1

            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

            out_path = os.path.join(OUTPUT_DIR, "image_result.jpg")
            cv2.imwrite(out_path, img)
            st.download_button(
                "⬇  EXPORT ANNOTATED IMAGE",
                open(out_path,"rb"), file_name="aird_result.jpg",
                use_container_width=True,
            )

        with col_info:
            section_divider("TARGET MANIFEST")

            if detected:
                for i, item in enumerate(detected):
                    if len(item) == 4:
                        label, conf, bbox, specs = item
                    else:
                        label, conf, bbox = item[:3]
                        specs = get_aircraft_specs(label)
                    target_card(label, conf, i+1, bbox, specs=specs)
            else:
                st.markdown("""
                <div style="font-family:'Share Tech Mono',monospace;font-size:10px;
                            letter-spacing:2px;color:rgba(240,234,216,.25);
                            padding:24px 0;text-align:center;">
                  NO TARGETS ACQUIRED
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            section_divider("SCAN SUMMARY")
            m1, m2 = st.columns(2)
            m1.metric("TARGETS", len(detected))
            avg = np.mean([c for _,c,_,_ in detected]) if detected else 0
            m2.metric("AVG CONF", f"{avg:.0%}" if detected else "—")

        if detected:
            section_divider("AIRCRAFT INTELLIGENCE DOSSIER", icon="🗎")
            if len(detected) == 1:
                target_item = detected[0]
                specs = target_item[3] if len(target_item) == 4 else get_aircraft_specs(target_item[0])
                render_intelligence_dossier(specs, title_prefix="TARGET #01")
            else:
                target_opts = [
                    f"Target #{i+1:02d} — {item[3].get('display_name', item[0]) if len(item)==4 else item[0]} ({item[1]:.0%})"
                    for i, item in enumerate(detected)
                ]
                sel_target_opt = st.selectbox(
                    "SELECT TARGET TO INSPECT DOSSIER",
                    options=target_opts,
                    key="img_dossier_sel"
                )
                sel_target_idx = target_opts.index(sel_target_opt)
                chosen_target = detected[sel_target_idx]
                specs = chosen_target[3] if len(chosen_target) == 4 else get_aircraft_specs(chosen_target[0])
                render_intelligence_dossier(specs, title_prefix=f"TARGET #{sel_target_idx+1:02d}")


# ─────────────────────────────────────────────
# MODE: VIDEO
# ─────────────────────────────────────────────
elif mode == "VIDEO":

    section_divider("SENSOR INPUT — VIDEO FEED")

    uploaded = st.file_uploader(
        "UPLOAD TARGET VIDEO", type=["mp4","avi","mov"],
        label_visibility="collapsed",
    )

    if not uploaded:
        empty_state("DROP AN MP4 · AVI · MOV FILE TO BEGIN FRAME ANALYSIS")

    else:
        section_divider("MISSION PARAMETERS")

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        cap   = cv2.VideoCapture(tfile.name)

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("RESOLUTION", f"{width}x{height}")
        c2.metric("FRAME RATE", f"{fps:.0f} FPS")
        c3.metric("TOTAL FRAMES", total)
        c4.metric("DURATION",   f"{total/fps:.1f}s")

        st.markdown("<br>", unsafe_allow_html=True)
        section_divider("PROCESSING PIPELINE")

        out_path = os.path.join(OUTPUT_DIR, "video_result.mp4")
        writer   = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height)
        )

        progress_bar = st.progress(0)
        status_text  = st.empty()
        frame_id, total_det = 0, 0
        tracker = TrackManager(max_history=40, max_stale_frames=30)
        geofence_mgr = GeofenceManager(config_path=GEOFENCES_PATH)
        geofence_alerts_log = []
        tracked_summary = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            ts_now = time.strftime("%H:%M:%S")

            # Run YOLO with ByteTrack Multi-Object Tracking
            results = detector.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=conf_thresh,
                iou=iou_thresh,
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

                    # Fine-grained CNN Classification
                    label, c_conf = classify_crop(crop, classifier)
                    track_state = tracker.update_track(
                        tid, (x1, y1, x2, y2), label, c_conf, float(d_conf), frame_id
                    )
                    total_det += 1

                    # Visual display uses smoothed label and confidence to eliminate flicker
                    display_label = track_state.smooth_label
                    display_conf = track_state.smooth_conf
                    track_specs = get_aircraft_specs(display_label)

                    # Geofencing entry evaluation (using tracked bounding box center)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    new_alerts = geofence_mgr.update_track(
                        track_id=tid,
                        center_point=(cx, cy),
                        label=display_label,
                        frame_idx=frame_id,
                        timestamp=ts_now
                    )
                    if new_alerts:
                        geofence_alerts_log.extend(new_alerts)

                    # Active breach check for this track
                    active_breaches = geofence_mgr.get_active_breaches().get(tid, [])
                    is_breaching = len(active_breaches) > 0
                    box_color = (20, 20, 230) if is_breaching else (10, 132, 212)

                    # Map zone IDs to zone names for display
                    breach_names = [
                        next((z.name for z in geofence_mgr.zones if z.id == zid), zid)
                        for zid in active_breaches
                    ]

                    # Movement Intelligence computation
                    move_info = get_movement_summary(track_state.trajectory, fps=fps, window=4)
                    tracked_summary[tid] = {
                        "label": display_label,
                        "conf": display_conf,
                        "bbox": (x1, y1, x2, y2),
                        "specs": track_specs,
                        "move_info": move_info,
                        "breaches": breach_names
                    }

                    if move_info["status"] == "MOVING":
                        move_text = f"{move_info['heading']} | {move_info['speed_px_sec']:.1f} px/s"
                    elif move_info["status"] == "INITIALIZING":
                        move_text = "INITIALIZING"
                    else:
                        move_text = "STATIONARY"

                    # Bounding box — tactical amber or alert red
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    # Corner brackets
                    L = 14
                    for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                        cv2.line(frame, (px, py), (px + dx * L, py), box_color, 2)
                        cv2.line(frame, (px, py), (px, py + dy * L), box_color, 2)

                    # Tactical Header Badge: TRACK #01 | Class | Conf (or RESTRICTED)
                    H_frame, W_frame = frame.shape[:2]
                    hud_text = f"TRACK #{tid:02d} | {display_label} | {display_conf:.0%}"
                    if is_breaching:
                        hud_text += " [RESTRICTED]"
                    (tw, th), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
                    bw, bh = tw + 16, th + 12
                    badge_bg = (20, 20, 90) if is_breaching else (46, 76, 28)
                    by1 = y1 - bh if y1 - bh >= 0 else y1
                    by2 = by1 + bh
                    bx1 = max(0, min(x1, W_frame - bw))
                    bx2 = min(W_frame, bx1 + bw)
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

            # Render restricted airspace geofence perimeters
            active_breaches_all = geofence_mgr.get_active_breaches()
            breached_zone_ids = [zid for zl in active_breaches_all.values() for zid in zl]
            geofence_mgr.draw_geofences(frame, show_labels=True, active_breach_zones=breached_zone_ids)

            # Render motion trajectory trails for all active tracks
            tracker.draw_trajectories(frame, color=(10, 132, 212), base_thickness=2)
            stale_ids = tracker.cleanup_stale(frame_id)
            geofence_mgr.cleanup_stale(stale_track_ids=stale_ids)

            ts = time.strftime("%H:%M:%S")
            alert_count = len(geofence_alerts_log)
            cv2.putText(frame, f"AIRD | {ts} | F:{frame_id:04d} | TRACKS:{tracker.total_unique_tracks:02d} | ALERTS:{alert_count:02d}",
                        (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (180, 160, 80), 1, cv2.LINE_AA)

            writer.write(frame)
            pct = frame_id / max(total, 1)
            progress_bar.progress(pct)
            status_text.markdown(f"""
            <div style="font-family:'Share Tech Mono',monospace;font-size:10px;
                        color:#8a9060;letter-spacing:2px;margin-top:4px;">
              FRAME {frame_id:04d}/{total:04d} &nbsp;·&nbsp;
              {pct:.0%} COMPLETE &nbsp;·&nbsp;
              {tracker.total_unique_tracks} UNIQUE TRACKS &nbsp;·&nbsp;
              {alert_count} GEOFENCE ALERTS &nbsp;·&nbsp;
              {total_det} DETECTIONS LOGGED
            </div>""", unsafe_allow_html=True)

        cap.release()
        writer.release()
        st.session_state.detections_total += total_det 
        st.session_state.frames_processed += frame_id

        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace;font-size:11px;
                    letter-spacing:2px;color:#7ab648;padding:10px 0;">
          MISSION COMPLETE — VIDEO ANALYSIS & MULTI-OBJECT TRACKING FINISHED
        </div>""", unsafe_allow_html=True)

        section_divider("PROCESSED OUTPUT")
        feed_header("VIDEO PLAYBACK WITH BYTETRACK MOTION TRAILS & GEOFENCING")
        st.video(out_path)
        st.download_button(
            "⬇  EXPORT ANNOTATED VIDEO",
            open(out_path,"rb"), file_name="aird_tracked_result.mp4",
            use_container_width=True,
        )

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("FRAMES SCANNED",   frame_id)
        r2.metric("UNIQUE TRACKS",    tracker.total_unique_tracks)
        r3.metric("TOTAL DETECTIONS", total_det)
        r4.metric("GEOFENCE ALERTS",  len(geofence_alerts_log))

        # Render Restricted-Zone Entry Alerts section
        render_geofence_alerts_summary(geofence_alerts_log)

        if tracked_summary:
            render_fleet_movement_summary(tracked_summary)

            section_divider("TRACKED FLEET TARGET CARDS & INTELLIGENCE", icon="🗎")
            col_tcards, col_tdossier = st.columns([1, 1], gap="large")

            with col_tcards:
                section_divider("TRACK MANIFEST", icon="▶")
                for tid, data in tracked_summary.items():
                    target_card(
                        label=data["label"],
                        conf=data["conf"],
                        index=tid,
                        bbox=data["bbox"],
                        track_id=tid,
                        specs=data["specs"],
                        movement_info=data["move_info"],
                        breaching_zones=data.get("breaches", [])
                    )

            with col_tdossier:
                section_divider("AIRCRAFT DOSSIER", icon="🗎")
                if len(tracked_summary) == 1:
                    t_id = list(tracked_summary.keys())[0]
                    t_data = tracked_summary[t_id]
                    render_intelligence_dossier(t_data["specs"], title_prefix=f"TRACK #{t_id:02d}")
                else:
                    t_opts = [
                        f"Track #{t_id:02d} — {data['specs'].get('display_name', data['label'])} ({data['conf']:.0%})"
                        for t_id, data in tracked_summary.items()
                    ]
                    sel_t_opt = st.selectbox(
                        "SELECT TRACKED AIRCRAFT DOSSIER",
                        options=t_opts,
                        key="vid_dossier_sel"
                    )
                    sel_t_idx = t_opts.index(sel_t_opt)
                    chosen_tid = list(tracked_summary.keys())[sel_t_idx]
                    chosen_data = tracked_summary[chosen_tid]
                    render_intelligence_dossier(chosen_data["specs"], title_prefix=f"TRACK #{chosen_tid:02d}")


# ─────────────────────────────────────────────
# MODE: LIVE CAMERA
# ─────────────────────────────────────────────
elif mode == "LIVE CAMERA":

    section_divider("LIVE SENSOR FEED — REAL-TIME TRACKING")

    st.markdown("""
    <div style="background:rgba(192,57,43,.08);
                border:1px solid rgba(192,57,43,.3);
                border-left:3px solid #c0392b;
                border-radius:4px;padding:10px 14px;margin-bottom:20px;">
      <span style="font-family:'Share Tech Mono',monospace;font-size:10px;
                   letter-spacing:2px;color:#e07060;">
        WARNING — LIVE MODE ACCESSES DEVICE CAMERA · GRANT PERMISSION IF PROMPTED
      </span>
    </div>
    """, unsafe_allow_html=True)

    col_ctrl, col_feed = st.columns([1, 3], gap="large")

    with col_ctrl:
        section_divider("CONTROLS")
        start       = st.checkbox("ACTIVATE SENSOR",  value=False)
        show_conf   = st.checkbox("SHOW CONFIDENCE",  value=True)
        show_corner = st.checkbox("CORNER BRACKETS",  value=True)
        show_trail  = st.checkbox("MOTION TRAILS",    value=True)
        show_zones  = st.checkbox("RESTRICTED ZONES", value=True)
        save_frames = st.checkbox("SAVE DETECTIONS",  value=False)

    with col_feed:
        feed_header("LIVE CAMERA STREAM WITH BYTETRACK & GEOFENCING")
        FRAME_WINDOW = st.image([], use_container_width=True)

        with st.expander("▶ TACTICAL PLATFORM SPECIFICATION LOOKUP", expanded=False):
            known_opts = [f"{get_aircraft_specs(c).get('display_name', c)} [{c}]" for c in CLASS_NAMES]
            sel_known = st.selectbox("SELECT PLATFORM DOSSIER", known_opts, key="live_specs_lookup")
            sel_cls = CLASS_NAMES[known_opts.index(sel_known)]
            render_intelligence_dossier(get_aircraft_specs(sel_cls))

        if start:
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.markdown("""
                <div style="font-family:'Share Tech Mono',monospace;font-size:11px;
                            color:#c0392b;letter-spacing:2px;padding:20px 0;">
                  CAMERA NOT DETECTED — CHECK DEVICE ACCESS
                </div>""", unsafe_allow_html=True)
            else:
                live_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                frame_count = 0
                tracker = TrackManager(max_history=40, max_stale_frames=30)
                geofence_mgr = GeofenceManager(config_path=GEOFENCES_PATH)

                while start:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    ts_now = time.strftime("%H:%M:%S")

                    # Run YOLO with ByteTrack tracking
                    results = detector.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=conf_thresh,
                        iou=iou_thresh,
                        verbose=False
                    )[0]

                    labels_frame = []
                    if results.boxes is not None and len(results.boxes) > 0:
                        boxes = results.boxes.xyxy.cpu().numpy()
                        det_confs = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else [1.0] * len(boxes)
                        track_ids = results.boxes.id.int().cpu().tolist() if results.boxes.id is not None else list(range(1, len(boxes) + 1))

                        for box, d_conf, tid in zip(boxes, det_confs, track_ids):
                            x1, y1, x2, y2 = map(int, box)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
                            crop = frame[y1:y2, x1:x2]
                            if crop.size == 0:
                                continue

                            label, conf = classify_crop(crop, classifier)
                            track_state = tracker.update_track(
                                tid, (x1, y1, x2, y2), label, conf, float(d_conf), frame_count
                            )
                            track_specs = get_aircraft_specs(track_state.smooth_label)

                            # Geofencing entry evaluation
                            if show_zones:
                                cx = int((x1 + x2) / 2)
                                cy = int((y1 + y2) / 2)
                                geofence_mgr.update_track(
                                    track_id=tid,
                                    center_point=(cx, cy),
                                    label=track_state.smooth_label,
                                    frame_idx=frame_count,
                                    timestamp=ts_now
                                )
                                active_breaches = geofence_mgr.get_active_breaches().get(tid, [])
                            else:
                                active_breaches = []

                            # Active breach check
                            is_breaching = len(active_breaches) > 0
                            box_color = (20, 20, 230) if is_breaching else (10, 132, 212)

                            # Movement Intelligence computation
                            move_info = get_movement_summary(track_state.trajectory, fps=live_fps, window=4)
                            if move_info["status"] == "MOVING":
                                move_text = f"{move_info['heading']} | {move_info['speed_px_sec']:.1f} px/s"
                            elif move_info["status"] == "INITIALIZING":
                                move_text = "INITIALIZING"
                            else:
                                move_text = "STATIONARY"

                            labels_frame.append((track_state.smooth_label, track_state.smooth_conf, tid, track_specs, move_info))

                            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                            if show_corner:
                                L = 14
                                for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                                    cv2.line(frame, (px, py), (px + dx * L, py), box_color, 2)
                                    cv2.line(frame, (px, py), (px, py + dy * L), box_color, 2)

                            H_frame, W_frame = frame.shape[:2]
                            txt = f"TRACK #{tid:02d} | {track_state.smooth_label}"
                            if show_conf:
                                txt += f" | {track_state.smooth_conf:.0%}"
                            if is_breaching:
                                txt += " [RESTRICTED]"

                            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
                            bw, bh = tw + 16, th + 12
                            badge_bg = (20, 20, 90) if is_breaching else (46, 76, 28)
                            by1 = y1 - bh if y1 - bh >= 0 else y1
                            by2 = by1 + bh
                            bx1 = max(0, min(x1, W_frame - bw))
                            bx2 = min(W_frame, bx1 + bw)
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

                    # Render restricted airspace geofence zones
                    if show_zones:
                        active_breaches_all = geofence_mgr.get_active_breaches()
                        breached_zone_ids = [zid for zl in active_breaches_all.values() for zid in zl]
                        geofence_mgr.draw_geofences(frame, show_labels=True, active_breach_zones=breached_zone_ids)

                    if show_trail:
                        tracker.draw_trajectories(frame, color=(10, 132, 212), base_thickness=2)
                    stale_ids = tracker.cleanup_stale(frame_count)
                    geofence_mgr.cleanup_stale(stale_track_ids=stale_ids)

                    ts = time.strftime("%H:%M:%S")
                    alert_count = len(geofence_mgr.alerts_history) if show_zones else 0
                    cv2.putText(frame, f"AIRD · {ts} · F:{frame_count:04d} · TRACKS:{tracker.total_unique_tracks:02d} · ALERTS:{alert_count:02d}",
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.42, (180, 160, 80), 1, cv2.LINE_AA)

                    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    if labels_frame:
                        st.session_state.detections_total += len(labels_frame)
                        st.session_state.frames_processed += 1
                        if save_frames:
                            fname = os.path.join(OUTPUT_DIR, f"frame_{frame_count:05d}.jpg")
                            cv2.imwrite(fname, frame)

                cap.release()


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="margin-top:40px;padding:14px 0;
            border-top:1px solid rgba(180,160,80,.15);
            display:flex;justify-content:space-between;align-items:center;">
  <span style="font-family:'Share Tech Mono',monospace;font-size:9px;
               letter-spacing:3px;color:rgba(240,234,216,.2);">
    AIRD · AIRCRAFT INTELLIGENCE RECOGNITION DASHBOARD
  </span>
  <span style="font-family:'Share Tech Mono',monospace;font-size:9px;
               letter-spacing:2px;color:#6b7c45;">
    POWERED BY YOLOv8 + TENSORFLOW CNN + BYTETRACK
  </span>
</div>
""", unsafe_allow_html=True)