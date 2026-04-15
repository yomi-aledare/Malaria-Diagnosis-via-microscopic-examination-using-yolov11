# importing required libraries
import cv2
from pathlib import Path
import os
import sys
import time
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BoboMal - Malaria Diagnosis App",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&family=Roboto:wght@300;400;500&display=swap');

:root {
    --blue:     #2986CC;
    --navy:     #0A2342;
    --cream:    #FFFDF7;
    --blue-dk:  #1a6aaa;
    --navy-lt:  #122d56;
    --blue-lt:  rgba(41,134,204,0.12);
}

html, body, [class*="css"] {
    font-family: 'Roboto', sans-serif;
    color: var(--navy);
}

.main { background-color: var(--cream) !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: 3px solid var(--blue);
}
section[data-testid="stSidebar"] * { color: var(--cream) !important; }

/* ── Sidebar logo ── */
.sidebar-logo {
    text-align: center;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid rgba(255,253,247,0.12);
    margin-bottom: 1.2rem;
}
.sidebar-logo h1 {
    font-family: 'Montserrat', sans-serif;
    font-weight: 800;
    font-size: 1.7rem;
    color: var(--blue) !important;
    margin: 0.4rem 0 0;
    letter-spacing: 1px;
}
.sidebar-logo p {
    font-size: 0.68rem;
    color: rgba(255,253,247,0.5) !important;
    margin: 0.2rem 0 0;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-family: 'Roboto', sans-serif;
}

/* ── Sidebar nav buttons — hide radio circles, style as pills ── */
div[data-testid="stRadio"] > div { gap: 0 !important; }
/* div[data-testid="stRadio"] [data-baseweb="radio"] { display: none !important; } */
div[data-testid="stRadio"] label {
    display: flex !important;
    align-items: center;
    gap: 0.7rem;
    width: 100%;
    padding: 0.7rem 1.1rem;
    border-radius: 10px;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    cursor: pointer;
    transition: all 0.2s;
    color: rgba(255,253,247,0.75) !important;
    margin-bottom: 0.25rem;
    background: transparent;
}
div[data-testid="stRadio"] label:hover {
    background: rgba(41,134,204,0.18) !important;
    color: white !important;
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: var(--blue) !important;
    color: white !important;
    box-shadow: 0 4px 14px rgba(41,134,204,0.35);
}

/* ── Page heading ── */
.page-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid var(--blue);
}
.page-header h2 {
    font-family: 'Montserrat', sans-serif;
    font-weight: 800;
    font-size: 1.8rem;
    color: var(--navy);
    margin: 0;
}
.page-header .badge {
    background: var(--blue);
    color: white;
    font-size: 0.68rem;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-family: 'Montserrat', sans-serif;
}

/* ── Feed zone ── */
.feed-zone {
    background: white;
    border: 2px dashed rgba(41,134,204,0.3);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.feed-zone:hover { border-color: var(--blue); }
.feed-zone .icon { font-size: 3rem; margin-bottom: 0.5rem; }
.feed-zone p { color: #777; font-size: 0.9rem; margin: 0; }

/* ── Result banners ── */
.result-neg {
    background: #e8f8f0;
    border-left: 5px solid #27ae60;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.result-pos {
    background: #fdecea;
    border-left: 5px solid #e74c3c;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.result-neg h4 { margin:0 0 3px; font-size:1rem; color:#27ae60; font-family:'Montserrat',sans-serif; }
.result-pos h4 { margin:0 0 3px; font-size:1rem; color:#e74c3c; font-family:'Montserrat',sans-serif; }
.result-neg p, .result-pos p { margin:0; font-size:0.82rem; color:#555; }

/* ── Summary card ── */
.summary-card {
    background: white;
    border-radius: 14px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(10,35,66,0.06);
}
.summary-card h4 {
    font-family: 'Montserrat', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: var(--navy);
    margin: 0 0 0.8rem;
    border-bottom: 1px solid #eee;
    padding-bottom: 0.5rem;
}
.stat-row { display:flex; justify-content:space-between; margin-bottom:0.45rem; font-size:0.88rem; }
.stat-row .val { font-weight:700; color:var(--blue); font-family:'Montserrat',sans-serif; }

/* ── About cards ── */
.about-card {
    background: white;
    border-radius: 18px;
    padding: 2.2rem 2.6rem;
    box-shadow: 0 8px 32px rgba(10,35,66,0.08);
    border-top: 5px solid var(--blue);
    margin-bottom: 1.5rem;
}
.about-card h2 {
    font-family: 'Montserrat', sans-serif;
    font-weight: 800;
    font-size: 1.8rem;
    color: var(--navy);
    margin-top: 0;
}
.about-card h3 {
    font-family: 'Montserrat', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: var(--navy);
    margin-top: 1.6rem;
    margin-bottom: 0.8rem;
}
.about-card p { line-height:1.8; color:#444; }
.tag {
    display: inline-block;
    background: var(--navy-lt);
    color: #FFFDF7 !important;
    font-size: 0.73rem;
    padding: 5px 13px;
    border-radius: 20px;
    margin: 3px;
    font-weight: 600;
    font-family: 'Montserrat', sans-serif;
}
.how-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 0.5rem;
}
.how-card {
    background: var(--blue-lt);
    border: 1px solid rgba(41,134,204,0.15);
    border-radius: 12px;
    padding: 1.2rem;
}
.how-card .step-icon { font-size:1.6rem; margin-bottom:0.4rem; }
.how-card strong {
    font-family: 'Montserrat', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    color: var(--navy);
    display: block;
    margin-bottom: 0.35rem;
}
.how-card p { font-size:0.83rem; color:#666; margin:0; line-height:1.6; }

/* ── Stat card ── */
.stat-card {
    background: white;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 4px 20px rgba(10,35,66,0.07);
    border-bottom: 4px solid var(--blue);
    text-align: center;
}
.stat-card .s-icon { font-size:2rem; }
.stat-card .s-val {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--navy);
    font-family: 'Montserrat', sans-serif;
    margin: 0.2rem 0 0;
}
.stat-card .s-label { font-size:0.82rem; color:#888; margin-top:2px; }

/* ── Login card ── */
.login-card {
    background: white;
    border-radius: 20px;
    padding: 2.8rem 3rem;
    box-shadow: 0 20px 60px rgba(10,35,66,0.12);
    border-top: 5px solid var(--blue);
}

/* ── Buttons ── */
.stButton > button {
    background: var(--blue) !important;
}

.stButton > button {
    background: var(--blue) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-family: 'Montserrat', sans-serif !important;
    letter-spacing: 0.5px !important;
    transition: background 0.2s, transform 0.1s !important;
}
.stButton > button:hover {
    background: var(--blue-dk) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Secondary buttons (mode toggles) */
.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: var(--blue) !important;
    border: 2px solid var(--blue) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: var(--blue-lt) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(41,134,204,0.3) !important;
    border-radius: 12px !important;
    padding: 0.8rem !important;
}

/* ── Inputs ── */
input[type="text"], input[type="password"] {
    border-radius: 10px !important;
    border: 2px solid #ddd !important;
    font-family: 'Roboto', sans-serif !important;
}
input:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(41,134,204,0.15) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; }
hr { border-color: rgba(255,253,247,0.12) !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
# Initialize session state variables for authentication, page routing, and diagnosis results

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "show_summary" not in st.session_state:
    st.session_state.show_summary = False
if "diagnosis_result" not in st.session_state:
    st.session_state.diagnosis_result = None
if "mode" not in st.session_state:
    st.session_state.mode = "Image"


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
def login_page():
    col_l, col_m, col_r = st.columns([1, 1.1, 1])
    with col_m:
        st.markdown("""
        <div style='text-align:center; padding:3rem 0 2rem;'>
            <div style='font-size:3.5rem;'>🔬</div>
            <h1 style='font-family:"Montserrat",sans-serif; font-weight:800;
                       color:#0A2342; font-size:2.4rem; margin:0.3rem 0 0; letter-spacing:1px;'>
                BoboMal
            </h1>
            <p style='color:#888; font-size:0.8rem; letter-spacing:2.5px;
                      text-transform:uppercase; margin-top:0.4rem; font-family:"Roboto",sans-serif;'>
                YOLO-Powered Blood Smear Analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", placeholder="Enter your password", type="password")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Login", use_container_width=True):
            if username == "admin" and password == "admin123":
                with st.spinner("Authenticating…"):
                    time.sleep(0.8)
                st.session_state.authenticated = True
                st.session_state.page = "Home"
                st.rerun()
            else:
                st.error("Invalid username or password.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <p style='text-align:center; color:#aaa; font-size:0.76rem; margin-top:1.5rem;
                  font-family:"Roboto",sans-serif;'>
            🔐Secure access · Medical use ONLY
        </p>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR  — radio buttons for page navigation + logout button
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <div style="font-size:2.2rem;">🔬</div>
            <h1>BoboMal</h1>
            <p>Malaria Diagnostic System</p>
        </div>
        """, unsafe_allow_html=True)

        pages = ["🏠Home", "🔬Diagnosis", "ℹ️About"]
        page_map = {"🏠Home": "Home", "🔬Diagnosis": "Diagnosis", "ℹ️About": "About"}
        reverse_map = {v: k for k, v in page_map.items()}

        current_idx = pages.index(reverse_map.get(st.session_state.page, "🏠Home"))

        nav = st.radio(
            "Navigation",
            pages,
            index=current_idx,
            label_visibility="collapsed",
        )
        st.session_state.page = page_map[nav]

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div style='padding:0.6rem 0 1rem; font-size:0.73rem;
                    opacity:0.35; text-align:center; font-family:"Roboto",sans-serif;'>
            BoboMal v1.0
        </div>
        """, unsafe_allow_html=True)

        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.diagnosis_result = None
            st.session_state.show_summary = False
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
def home_page():
    st.markdown("""
    <div class="page-header">
        <h2>Dashboard</h2>
        <span>Welcome,</span> <span class="badge">Admin</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    for col, icon, label, val in [
        (c1, "🧫", "Average Inference Time", "~2 secs"),
        (c2, "⚠️", "Required Image Size", "640px"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="s-icon">{icon}</div>
                <div class="s-val">{val}</div>
                <div class="s-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:linear-gradient(135deg,#0A2342 0%,#122d56 100%);
                border-radius:18px; padding:2.2rem 2.8rem; color:#FFFDF7;'>
        <h3 style='font-family:"Montserrat",sans-serif; font-weight:800;
                   color:#2986CC; margin-top:0; font-size:1.4rem;'>
            Welcome to BoboMal
        </h3>
        <p style='line-height:1.8; opacity:0.85; max-width:680px;
                  font-family:"Roboto",sans-serif; font-size:0.95rem;'>
            This system uses <strong style='color:#2986CC;'>YOLOv11 object detection</strong>
            on Giemsa-stained blood smear images to identify <em>Plasmodium</em> parasites
            in real time. Navigate to the <strong>Diagnosis</strong> page to upload a sample
            and receive an instant analysis.
        </p>
        <div style='margin-top:1.2rem; display:flex; gap:0.6rem; flex-wrap:wrap;'>
            <span style='background:rgba(41,134,204,0.22); color:#7ec8f7; padding:5px 14px;
                         border-radius:20px; font-size:0.78rem; font-weight:700;
                         font-family:"Montserrat",sans-serif;'>YOLOv11</span>
            <span style='background:rgba(41,134,204,0.22); color:#7ec8f7; padding:5px 14px;
                         border-radius:20px; font-size:0.78rem; font-weight:700;
                         font-family:"Montserrat",sans-serif;'>Blood Smear</span>
            <span style='background:rgba(41,134,204,0.22); color:#7ec8f7; padding:5px 14px;
                         border-radius:20px; font-size:0.78rem; font-weight:700;
                         font-family:"Montserrat",sans-serif;'>Malaria Detection</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSIS PAGE  — mode selector on this page (not sidebar)
# ══════════════════════════════════════════════════════════════════════════════
def diagnosis_page():
    st.markdown("""
    <div class="page-header">
        <h2>Diagnosis</h2>
        <span class="badge">Analysis</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Mode toggle (Image / Video) ───────────────────────────────────────────
    mode_col1, mode_col2, _ = st.columns([0.9, 0.9, 5])
    with mode_col1:
        img_type = "primary" if st.session_state.mode == "Image" else "secondary"
        if st.button("🖼  Image", key="mode_img", use_container_width=True, type=img_type):
            st.session_state.mode = "Image"
            st.rerun()
    with mode_col2:
        vid_type = "primary" if st.session_state.mode == "Video" else "secondary"
        if st.button("🎥  Video", key="mode_vid", use_container_width=True, type=vid_type):
            st.session_state.mode = "Video"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1.1, 1], gap="large")

    # ── Upload / Feed ──────────────────────────────────────────────────────────
    with left_col:
        st.markdown(f"""
        <div style='font-size:0.73rem; font-weight:700; letter-spacing:1.5px;
                    text-transform:uppercase; color:#999; margin-bottom:0.6rem;
                    font-family:"Montserrat",sans-serif;'>
            {'🖼  Image Feed' if st.session_state.mode == 'Image' else '🎥  Video Feed'}
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.mode == "Image":
            uploaded = st.file_uploader(
                "Upload blood smear image",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                label_visibility="collapsed",
            )
            if uploaded:
                img = Image.open(uploaded)
                st.image(img, use_container_width=True,
                         caption="Uploaded smear — ready for analysis")
                st.session_state.uploaded_file = uploaded
            else:
                st.markdown("""
                <div class="feed-zone">
                    <div class="icon">🧬</div>
                    <p>Drag &amp; drop a blood smear image<br>
                    <small>JPG · PNG · BMP · TIFF</small></p>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.uploaded_file = None
        else:
            st.markdown("""
            <div class="feed-zone">
                <div class="icon">🎥</div>
                <p>Live camera feed or video upload<br>
                <small>Connect a microscope camera or upload MP4</small></p>
            </div>
            """, unsafe_allow_html=True)
            uploaded_vid = st.file_uploader(
                "Upload video",
                type=["mp4", "avi", "mov"],
                label_visibility="collapsed",
            )
            if uploaded_vid:
                st.video(uploaded_vid)

    # ── Results panel ─────────────────────────────────────────────────────────
    with right_col:
        st.markdown("""
        <div style='font-size:0.73rem; font-weight:700; letter-spacing:1.5px;
                    text-transform:uppercase; color:#999; margin-bottom:0.6rem;
                    font-family:"Montserrat",sans-serif;'>
            🔍 Analysis Result
        </div>
        """, unsafe_allow_html=True)

        run_btn = st.button("▶  Run Analysis", use_container_width=True)

        if run_btn:
            with st.spinner("Running YOLO inference…"):
                time.sleep(1.8)   # ← replace with your model call
                
                
                
            st.session_state.diagnosis_result = (
                "negative" if st.session_state.diagnosis_result == "positive" else "positive"
            )
            st.session_state.show_summary = False

        if st.session_state.diagnosis_result == "negative":
            st.markdown("""
            <div class="result-neg">
                <div style='font-size:2.2rem;'>✅</div>
                <div>
                    <h4>Negative — No Parasite Detected</h4>
                    <p>No <em>Plasmodium</em> parasites were identified in the smear.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif st.session_state.diagnosis_result == "positive":
            st.markdown("""
            <div class="result-pos">
                <div style='font-size:2.2rem;'>⚠️</div>
                <div>
                    <h4>Positive — Parasite Detected</h4>
                    <p><em>Plasmodium</em> parasites detected. Refer for clinical confirmation.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#f5f7fa; border-radius:12px; padding:1.4rem;
                        text-align:center; color:#bbb; border:1px dashed #ddd;'>
                <div style='font-size:2rem;'>🔬</div>
                <p style='margin:0; font-size:0.88rem; font-family:"Roboto",sans-serif;'>
                    Upload a sample and run analysis to see results here.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.diagnosis_result:
            if st.button("📋  Show Summary", use_container_width=True):
                st.session_state.show_summary = not st.session_state.show_summary

            if st.session_state.show_summary:
                result_label = "Negative ✅" if st.session_state.diagnosis_result == "negative" else "Positive ⚠️"
                result_color = "#27ae60" if st.session_state.diagnosis_result == "negative" else "#e74c3c"
                st.markdown(f"""
                <div class="summary-card" style='margin-top:0.8rem;'>
                    <h4>Analysis Summary</h4>
                    <div class="stat-row">
                        <span>Diagnosis</span>
                        <span class="val" style="color:{result_color};">{result_label}</span>
                    </div>
                    <div class="stat-row">
                        <span>Model</span>
                        <span class="val">YOLOv11</span>
                    </div>
                    <div class="stat-row">
                        <span>Confidence</span>
                        <span class="val">91.4%</span>
                    </div>
                    <div class="stat-row">
                        <span>Specificity</span>
                        <span class="val">38</span>
                    </div>
                    <div class="stat-row">
                        <span>Inference Time</span>
                        <span class="val">1.8 s</span>
                    </div>
                    <div class="stat-row">
                        <span>Mode</span>
                        <span class="val">{st.session_state.mode}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABOUT PAGE  — static content describing the project, tech stack, and how it works
# ══════════════════════════════════════════════════════════════════════════════
def about_page():
    st.markdown("""
    <div class="page-header">
        <h2>About</h2>
        <span class="badge">Project</span>
    </div>
    """, unsafe_allow_html=True)

    # Card 1 — Project description + tech tags
    st.markdown("""
    <div class="about-card">
        <h2>🔬 BoboMal</h2>
        <p>
            This project uses <strong>digital blood smear images</strong> to diagnose
            malaria parasites using <strong>YOLO (You Only Look Once) object detection</strong>.
            Giemsa-stained thin blood smears are analysed in real time, enabling fast,
            scalable, and accurate identification of <em>Plasmodium</em> species without
            requiring specialist laboratory personnel at the point of care.
        </p>
        <p>
            The YOLO model is trained on annotated microscopy images and can distinguish
            between infected and healthy red blood cells at the cell level, producing
            confidence scores and bounding-box overlays for clinical review.
        </p>
        <h3>Key Technologies</h3>
        <div>
            <span class="tag">YOLOv11</span>
            <span class="tag">Python</span>
            <span class="tag">Streamlit</span>
            <span class="tag">OpenCV</span>
            <span class="tag">PyTorch</span>
            <span class="tag">Blood Smear Analysis</span>
            <span class="tag">Object Detection</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Card 2 — How It Works grid
    st.markdown("""
    <div class="about-card">
        <h3 style="margin-top:0;">How It Works</h3>
        <div class="how-grid">
            <div class="how-card">
                <div class="step-icon">🩸</div>
                <strong>1. Sample Input</strong>
                <p>Upload a Giemsa-stained thin blood smear image or connect a live microscope feed.</p>
            </div>
            <div class="how-card">
                <div class="step-icon">🧠</div>
                <strong>2. YOLO Inference</strong>
                <p>YOLOv11 scans the smear at cell level, detecting parasite morphology in milliseconds.</p>
            </div>
            <div class="how-card">
                <div class="step-icon">📊</div>
                <strong>3. Result &amp; Confidence</strong>
                <p>A diagnosis (Positive / Negative) is returned with a confidence score and bounding boxes.</p>
            </div>
            <div class="how-card">
                <div class="step-icon">📋</div>
                <strong>4. Clinical Review</strong>
                <p>Summary report generated for clinician review and patient referral if needed.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL INFERENCE FUNCTION  —  YOLO model inference code
# ══════════════════════════════════════════════════════════════════════════════
def run_yolo_inference(image):
    # Placeholder function — replace with actual YOLO model inference code
    # Example:
    # model = YOLO("path/to/your/yolo-model.pt")
    # results = model(image)
    # Process results to determine diagnosis and confidence
    time.sleep(1.8)  # Simulate inference time
    return "positive", 0.914  # Example output: (diagnosis, confidence)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.authenticated:
    login_page()
else:
    render_sidebar()
    page = st.session_state.page
    if page == "Home":
        home_page()
    elif page == "Diagnosis":
        diagnosis_page()
    elif page == "About":
        about_page()
