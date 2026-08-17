import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ─────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DETECTOR_PATH   = os.path.join(PROJECT_ROOT, "runs", "detect", "AircraftDetector3", "weights", "best.pt")
CLASSIFIER_PATH = os.path.join(PROJECT_ROOT, "models", "aircraft_classifier.h5")
CLASS_NAMES     = sorted(os.listdir(os.path.join(PROJECT_ROOT, "dataset", "classifier_dataset", "train")))
OUTPUT_DIR      = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
            TACTICAL AI RECOGNITION DASHBOARD · v2.0
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
            MDL LOADED
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


def target_card(label, conf, index, bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    conf_bar  = int(conf * 100)
    conf_color = "#7ab648" if conf > 0.7 else "#f5a623" if conf > 0.45 else "#c0392b"
    st.markdown(f"""
    <div style="background:#1e2416;border:1px solid rgba(180,160,80,.2);
                border-left:3px solid {conf_color};border-radius:4px;
                padding:12px 14px;margin-bottom:8px;
                font-family:'Share Tech Mono',monospace;">
      <div style="display:flex;justify-content:space-between;
                  align-items:center;margin-bottom:10px;">
        <span style="font-family:'Bebas Neue',sans-serif;font-size:17px;
                     letter-spacing:2px;color:#f0ead8;">{label}</span>
        <span style="font-size:9px;letter-spacing:2px;color:#6b7c45;
                     background:#2e3a1c;padding:3px 8px;border-radius:3px;">
          TGT {index:02d}
        </span>
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
      <span>CONF THRESHOLD ACTIVE · YOLOv8 + CNN</span>
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
        ("DETECTOR",   "YOLOv8n",           90, "#7ab648"),
        ("CLASSIFIER", "CNN / KERAS",        85, "#7ab648"),
        ("CLASSES",    str(len(CLASS_NAMES)),min(len(CLASS_NAMES),100), "#f5a623"),
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
            feed_header()

            with st.spinner("RUNNING DETECTION PIPELINE..."):
                img     = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)
                results = detector(img, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]

                detected = []
                for box in results.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    label, conf = classify_crop(crop, classifier)
                    detected.append((label, conf, (x1, y1, x2, y2)))

                    # Bounding box — amber in BGR
                    cv2.rectangle(img, (x1,y1), (x2,y2), (10,132,212), 2)
                    # Corner brackets
                    L = 16
                    for (px,py,dx,dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                        cv2.line(img,(px,py),(px+dx*L,py),(10,132,212),2)
                        cv2.line(img,(px,py),(px,py+dy*L),(10,132,212),2)
                    # Label
                    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                    cv2.rectangle(img,(x1,y1-22),(x1+tw+12,y1),(46,76,28),-1)
                    cv2.rectangle(img,(x1,y1-22),(x1+tw+12,y1),(10,132,212),1)
                    cv2.putText(img, label,(x1+6,y1-6),
                                cv2.FONT_HERSHEY_SIMPLEX,0.52,(240,234,216),1,cv2.LINE_AA)
                    cv2.putText(img,f"{conf:.0%}",(x2+4,y1+14),
                                cv2.FONT_HERSHEY_SIMPLEX,0.40,(72,182,122),1,cv2.LINE_AA)

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
                for i, (label, conf, bbox) in enumerate(detected):
                    target_card(label, conf, i+1, bbox)
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
            avg = np.mean([c for _,c,_ in detected]) if detected else 0
            m2.metric("AVG CONF", f"{avg:.0%}" if detected else "—")


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

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            results = detector(frame, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
            for box in results.boxes.xyxy:
                x1,y1,x2,y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                label, conf = classify_crop(crop, classifier)
                total_det += 1

                cv2.rectangle(frame,(x1,y1),(x2,y2),(10,132,212),2)
                L = 14
                for (px,py,dx,dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                    cv2.line(frame,(px,py),(px+dx*L,py),(10,132,212),2)
                    cv2.line(frame,(px,py),(px,py+dy*L),(10,132,212),2)
                (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.50,1)
                cv2.rectangle(frame,(x1,y1-20),(x1+tw+10,y1),(46,76,28),-1)
                cv2.putText(frame,label,(x1+5,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.50,(240,234,216),1,cv2.LINE_AA)
                cv2.putText(frame,f"{conf:.0%}",(x2+4,y1+12),
                            cv2.FONT_HERSHEY_SIMPLEX,0.38,(72,182,122),1,cv2.LINE_AA)

            ts = time.strftime("%H:%M:%S")
            cv2.putText(frame,f"AIRD | {ts} | F:{frame_id:04d} | DET:{total_det:03d}",
                        (10,height-10),cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,(180,160,80),1,cv2.LINE_AA)

            writer.write(frame)
            pct = frame_id / max(total, 1)
            progress_bar.progress(pct)
            status_text.markdown(f"""
            <div style="font-family:'Share Tech Mono',monospace;font-size:10px;
                        color:#8a9060;letter-spacing:2px;margin-top:4px;">
              FRAME {frame_id:04d}/{total:04d} &nbsp;·&nbsp;
              {pct:.0%} COMPLETE &nbsp;·&nbsp;
              {total_det} DETECTIONS LOGGED
            </div>""", unsafe_allow_html=True)

        cap.release()
        writer.release()
        st.session_state.detections_total += total_det 
        st.session_state.frames_processed += frame_id

        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace;font-size:11px;
                    letter-spacing:2px;color:#7ab648;padding:10px 0;">
          MISSION COMPLETE — VIDEO ANALYSIS FINISHED
        </div>""", unsafe_allow_html=True)

        section_divider("PROCESSED OUTPUT")
        feed_header("VIDEO PLAYBACK")
        st.video(out_path)
        st.download_button(
            "⬇  EXPORT ANNOTATED VIDEO",
            open(out_path,"rb"), file_name="aird_result.mp4",
            use_container_width=True,
        )

        r1,r2,r3 = st.columns(3)
        r1.metric("FRAMES SCANNED",   frame_id)
        r2.metric("TOTAL DETECTIONS", total_det)
        r3.metric("DET / FRAME",      f"{total_det/max(frame_id,1):.2f}")


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
        save_frames = st.checkbox("SAVE DETECTIONS",  value=False)

    with col_feed:
        feed_header("LIVE CAMERA STREAM")
        FRAME_WINDOW = st.image([], use_container_width=True)

        if start:
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.markdown("""
                <div style="font-family:'Share Tech Mono',monospace;font-size:11px;
                            color:#c0392b;letter-spacing:2px;padding:20px 0;">
                  CAMERA NOT DETECTED — CHECK DEVICE ACCESS
                </div>""", unsafe_allow_html=True)
            else:
                frame_count = 0
                while start:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1

                    results = detector(frame, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
                    labels_frame = []

                    for box in results.boxes.xyxy:
                        x1,y1,x2,y2 = map(int, box)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        label, conf = classify_crop(crop, classifier)
                        labels_frame.append((label, conf))

                        cv2.rectangle(frame,(x1,y1),(x2,y2),(10,132,212),2)
                        if show_corner:
                            L = 14
                            for (px,py,dx,dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                                cv2.line(frame,(px,py),(px+dx*L,py),(10,132,212),2)
                                cv2.line(frame,(px,py),(px,py+dy*L),(10,132,212),2)
                        txt = f"{label} {conf:.0%}" if show_conf else label
                        (tw,th),_ = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.50,1)
                        cv2.rectangle(frame,(x1,y1-20),(x1+tw+10,y1),(46,76,28),-1)
                        cv2.putText(frame,txt,(x1+5,y1-5),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.50,(240,234,216),1,cv2.LINE_AA)

                    ts = time.strftime("%H:%M:%S")
                    cv2.putText(frame,f"AIRD · {ts} · F:{frame_count:04d}",
                                (10,20),cv2.FONT_HERSHEY_SIMPLEX,
                                0.42,(180,160,80),1,cv2.LINE_AA)

                    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    if labels_frame:
                        st.session_state.detections_total += len(labels_frame)
                        st.session_state.frames_processed += 1
                        if save_frames:
                            fname = os.path.join(OUTPUT_DIR,f"frame_{frame_count:05d}.jpg")
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
    POWERED BY YOLOv8 + TENSORFLOW CNN
  </span>
</div>
""", unsafe_allow_html=True)