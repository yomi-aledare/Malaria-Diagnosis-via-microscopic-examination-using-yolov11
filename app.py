#importing required libraries
import cv2
from pathlib import Path
import os
import sys
import time
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Set page configuration
st.set_page_config(
    page_title="Malaria Diagnosis App",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS styles ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');
 
:root {
    --coral:    #FF7F50;
    --navy:     #0A2342;
    --cream:    #EFE9E7;
    --coral-dk: #e06030;
    --navy-lt:  #122d56;
    --glass:    rgba(10,35,66,0.07);
}
 
/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--navy);
}
 
.main { background-color: var(--cream); }
section[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: 3px solid var(--coral);
}
section[data-testid="stSidebar"] * { color: var(--cream) !important; }
 
/* ── Sidebar logo block ── */
.sidebar-logo {
    text-align: center;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid rgba(239,233,231,0.15);
    margin-bottom: 1.5rem;
}
.sidebar-logo h1 {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    color: var(--coral) !important;
    margin: 0;
    letter-spacing: 0.5px;
}
.sidebar-logo p {
    font-size: 0.72rem;
    color: rgba(239,233,231,0.55) !important;
    margin: 0.2rem 0 0;
    letter-spacing: 2px;
    text-transform: uppercase;
}
 
/* ── Sidebar nav pills ── */
div[data-testid="stRadio"] label {
    display: block;
    padding: 0.55rem 1.1rem;
    border-radius: 8px;
    font-size: 0.92rem;
    font-weight: 500;
    letter-spacing: 0.3px;
    cursor: pointer;
    transition: background 0.2s;
}
div[data-testid="stRadio"] label:hover { background: rgba(255,127,80,0.15); }
div[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) + div {
    background: var(--coral);
    border-radius: 8px;
}
 
/* ── Login card ── */
.login-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 80vh;
}
.login-card {
    background: white;
    border-radius: 20px;
    padding: 3rem 3.5rem;
    width: 100%;
    max-width: 440px;
    box-shadow: 0 20px 60px rgba(10,35,66,0.12);
    border-top: 5px solid var(--coral);
}
.login-card h2 {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: var(--navy);
    margin-bottom: 0.2rem;
}
.login-card .sub {
    color: #888;
    font-size: 0.85rem;
    margin-bottom: 2rem;
}
 
/* ── Page heading ── */
.page-header {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid var(--coral);
}
.page-header h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    color: var(--navy);
    margin: 0;
}
.page-header .badge {
    background: var(--coral);
    color: white;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}
 
/* ── Feed / upload zone ── */
.feed-zone {
    background: white;
    border: 2px dashed rgba(10,35,66,0.2);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.feed-zone:hover { border-color: var(--coral); }
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
.result-neg h4, .result-pos h4 { margin: 0 0 3px; font-size: 1rem; }
.result-neg p,  .result-pos p  { margin: 0; font-size: 0.82rem; color: #555; }
 
/* ── Summary card ── */
.summary-card {
    background: white;
    border-radius: 14px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(10,35,66,0.06);
}
.summary-card h4 {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: var(--navy);
    margin: 0 0 0.8rem;
    border-bottom: 1px solid #eee;
    padding-bottom: 0.5rem;
}
.stat-row { display: flex; justify-content: space-between; margin-bottom: 0.45rem; font-size: 0.88rem; }
.stat-row .val { font-weight: 600; color: var(--coral); }
 
/* ── About card ── */
.about-card {
    background: white;
    border-radius: 18px;
    padding: 2.5rem 3rem;
    box-shadow: 0 8px 32px rgba(10,35,66,0.08);
    border-top: 5px solid var(--coral);
}
.about-card h2 {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: var(--navy);
    margin-top: 0;
}
.about-card p { line-height: 1.75; color: #444; }
.tag {
    display: inline-block;
    background: var(--navy);
    color: var(--cream) !important;
    font-size: 0.75rem;
    padding: 4px 12px;
    border-radius: 20px;
    margin: 3px;
    font-weight: 500;
}
 
/* ── Primary button override ── */
.stButton > button {
    background: var(--coral) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.5px !important;
    transition: background 0.2s, transform 0.1s !important;
}
.stButton > button:hover {
    background: var(--coral-dk) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
 
/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(10,35,66,0.25) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
 
/* ── Inputs ── */
input[type="text"], input[type="password"] {
    border-radius: 10px !important;
    border: 2px solid #ddd !important;
    font-family: 'DM Sans', sans-serif !important;
}
input:focus {
    border-color: var(--coral) !important;
    box-shadow: 0 0 0 3px rgba(255,127,80,0.15) !important;
}
 
/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state initialization ──
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "show_summary" not in st.session_state:
    st.session_state.show_summary = False
if "diagnosis_result" not in st.session_state:
    st.session_state.diagnosis_result = None   # None | "positive" | "negative"
if "mode" not in st.session_state:
    st.session_state.mode = "Image"


# ── LOGIN PAGE ──
# ══════════════════════════════════════════════════════════════════════════════
def login_page():
    col_l, col_m, col_r = st.columns([1, 1.2, 1])
    with col_m:
        st.markdown("""
        <div style='text-align:center; padding: 3rem 0 2rem;'>
            <div style='font-size:3.5rem;'>🔬</div>
            <h1 style='font-family:"Playfair Display",serif; color:#0A2342;
                       font-size:2.4rem; margin:0.3rem 0 0;'>MalariaScope</h1>
            <p style='color:#888; font-size:0.85rem; letter-spacing:2px;
                      text-transform:uppercase; margin-top:0.3rem;'>
                YOLO-Powered Blood Smear Analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
 
        with st.container():
            st.markdown("""
            <div style='background:white; border-radius:20px; padding:2.5rem 2.8rem;
                        box-shadow:0 20px 60px rgba(10,35,66,0.12);
                        border-top:5px solid #FF7F50;'>
            """, unsafe_allow_html=True)
 
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", placeholder="Enter your password", type="password")
            st.markdown("<br>", unsafe_allow_html=True)
 
            if st.button("🔐  Login", use_container_width=True):
                if username =="admin" and password =="admin123":
                    with st.spinner("Authenticating…"):
                        time.sleep(0.8)
                    st.session_state.authenticated = True
                    st.session_state.page = "Home"
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
 
            st.markdown("</div>", unsafe_allow_html=True)
 
        st.markdown("""
        <p style='text-align:center; color:#aaa; font-size:0.78rem; margin-top:2rem;'>
            Secure access · Medical use only
        </p>
        """, unsafe_allow_html=True)
 
 # SIDEBAR (post-login)
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <div style="font-size:2rem;">🔬</div>
            <h1>MalariaScope</h1>
            <p>Diagnostic System</p>
        </div>
        """, unsafe_allow_html=True)
 
        nav = st.radio(
            "Navigation",
            ["🏠  Home", "🔬  Diagnosis", "ℹ️  About"],
            label_visibility="collapsed",
            index=["🏠  Home", "🔬  Diagnosis", "ℹ️  About"].index(
                {"Home": "🏠  Home", "Diagnosis": "🔬  Diagnosis", "About": "ℹ️  About"}
                .get(st.session_state.page, "🏠  Home")
            )
        )
        st.session_state.page = {"🏠  Home": "Home", "🔬  Diagnosis": "Diagnosis", "ℹ️  About": "About"}[nav]
 
        # Mode selector shown only on Diagnosis page
        if st.session_state.page == "Diagnosis":
            st.markdown("---")
            st.markdown("<p style='font-size:0.8rem; opacity:0.6; letter-spacing:1px; text-transform:uppercase;'>Mode</p>",
                        unsafe_allow_html=True)
            mode = st.radio("Mode", ["🖼  Image", "🎥  Video"],
                            label_visibility="collapsed",
                            index=0 if st.session_state.mode == "Image" else 1)
            st.session_state.mode = "Image" if "Image" in mode else "Video"
 
        st.markdown("---")
        st.markdown("""
        <div style='padding: 1rem 0; font-size:0.78rem; opacity:0.4; text-align:center;'>
            DetectMalApp v1.0<br>YOLO Object Detection
        </div>
        """, unsafe_allow_html=True)
 
        if st.button("🚪  Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.diagnosis_result = None
            st.session_state.show_summary = False
            st.rerun()
 
 
 # HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
def home_page():
    st.markdown("""
    <div class="page-header">
        <h2>Dashboard</h2>
        <span class="badge">Live</span>
    </div>
    """, unsafe_allow_html=True)
 
    c1, c2, c3 = st.columns(3)
    for col, icon, label, val in [
        (c1, "🧫", "Samples Analysed", "1,248"),
        (c2, "✅", "Negative Cases",   "1,031"),
        (c3, "⚠️", "Positive Cases",   "217"),
    ]:
        with col:
            st.markdown(f"""
            <div style='background:white; border-radius:16px; padding:1.6rem 1.8rem;
                        box-shadow:0 4px 20px rgba(10,35,66,0.07);
                        border-bottom:4px solid #FF7F50; text-align:center;'>
                <div style='font-size:2.2rem;'>{icon}</div>
                <div style='font-size:1.9rem; font-weight:700; color:#0A2342;
                            font-family:"Playfair Display",serif;'>{val}</div>
                <div style='font-size:0.82rem; color:#888; margin-top:2px;'>{label}</div>
            </div>
            """, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0A2342,#122d56);
                border-radius:18px; padding:2.2rem 2.8rem; color:#EFE9E7;'>
        <h3 style='font-family:"Playfair Display",serif; color:#FF7F50;
                   margin-top:0;'>Welcome to MalariaScope</h3>
        <p style='line-height:1.75; opacity:0.85; max-width:680px;'>
            This system uses <strong style='color:#FF7F50;'>YOLOv8 object detection</strong>
            on Giemsa-stained blood smear images to identify <em>Plasmodium</em> parasites
            in real time. Navigate to the <strong>Diagnosis</strong> page to upload a sample
            and receive an instant analysis.
        </p>
        <div style='margin-top:1.2rem; display:flex; gap:0.6rem; flex-wrap:wrap;'>
            <span style='background:rgba(255,127,80,0.2); color:#FF7F50; padding:4px 14px;
                         border-radius:20px; font-size:0.8rem; font-weight:600;'>YOLOv8</span>
            <span style='background:rgba(255,127,80,0.2); color:#FF7F50; padding:4px 14px;
                         border-radius:20px; font-size:0.8rem; font-weight:600;'>Blood Smear</span>
            <span style='background:rgba(255,127,80,0.2); color:#FF7F50; padding:4px 14px;
                         border-radius:20px; font-size:0.8rem; font-weight:600;'>Malaria Detection</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
 # DIAGNOSIS PAGE
# ══════════════════════════════════════════════════════════════════════════════
def diagnosis_page():
    st.markdown("""
    <div class="page-header">
        <h2>Diagnosis</h2>
        <span class="badge">Analysis</span>
    </div>
    """, unsafe_allow_html=True)
 
    left_col, right_col = st.columns([1.1, 1], gap="large")
 
    # ── Upload / Feed ──────────────────────────────────────────────────────────
    with left_col:
        st.markdown(f"""
        <div style='font-size:0.8rem; font-weight:600; letter-spacing:1px;
                    text-transform:uppercase; color:#888; margin-bottom:0.6rem;'>
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
                    <p>Drag & drop a blood smear image<br>
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
        <div style='font-size:0.8rem; font-weight:600; letter-spacing:1px;
                    text-transform:uppercase; color:#888; margin-bottom:0.6rem;'>
            🔍 Analysis Result
        </div>
        """, unsafe_allow_html=True)
 
        run_btn = st.button("▶  Run Analysis", use_container_width=True)
 
        if run_btn:
            with st.spinner("Running YOLO inference…"):
                time.sleep(1.8)   # ← replace with your model call
            # Demo: toggle result for demonstration
            st.session_state.diagnosis_result = (
                "negative" if st.session_state.diagnosis_result == "positive" else "positive"
            )
            st.session_state.show_summary = False
 
        if st.session_state.diagnosis_result == "negative":
            st.markdown("""
            <div class="result-neg">
                <div style='font-size:2.2rem;'>✅</div>
                <div>
                    <h4 style='color:#27ae60;'>Negative — No Parasite Detected</h4>
                    <p>No <em>Plasmodium</em> parasites were identified in the smear.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
 
        elif st.session_state.diagnosis_result == "positive":
            st.markdown("""
            <div class="result-pos">
                <div style='font-size:2.2rem;'>⚠️</div>
                <div>
                    <h4 style='color:#e74c3c;'>Positive — Parasite Detected</h4>
                    <p><em>Plasmodium</em> parasites detected. Refer for clinical confirmation.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#f5f5f5; border-radius:12px; padding:1.4rem;
                        text-align:center; color:#aaa;'>
                <div style='font-size:2rem;'>🔬</div>
                <p style='margin:0; font-size:0.9rem;'>
                    Upload a sample and run analysis to see results here.
                </p>
            </div>
            """, unsafe_allow_html=True)
 
        st.markdown("<br>", unsafe_allow_html=True)
 
        # ── Show Summary ──
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
                        <span class="val">YOLOv8</span>
                    </div>
                    <div class="stat-row">
                        <span>Confidence</span>
                        <span class="val">91.4%</span>
                    </div>
                    <div class="stat-row">
                        <span>Cells Scanned</span>
                        <span class="val">3,842</span>
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
 
 # ABOUT PAGE
# ══════════════════════════════════════════════════════════════════════════════
def about_page():
    st.markdown("""
    <div class="page-header">
        <h2>About</h2>
        <span class="badge">Project</span>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("""
    <div class="about-card">
        <h2>🔬 MalariaScope</h2>
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
 
        <h3 style='font-family:"Playfair Display",serif; color:#0A2342; margin-top:2rem;'>
            Key Technologies
        </h3>
        <div>
            <span class="tag">YOLOv8</span>
            <span class="tag">Python</span>
            <span class="tag">Streamlit</span>
            <span class="tag">OpenCV</span>
            <span class="tag">PyTorch</span>
            <span class="tag">Blood Smear Analysis</span>
            <span class="tag">Object Detection</span>
        </div>
 
        <h3 style='font-family:"Playfair Display",serif; color:#0A2342; margin-top:2rem;'>
            How It Works
        </h3>
        <div style='display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:1rem;'>
            <div style='background:#f8f8f8; border-radius:12px; padding:1.2rem;'>
                <div style='font-size:1.5rem;'>🩸</div>
                <strong>1. Sample Input</strong>
                <p style='font-size:0.85rem; color:#666; margin:0.3rem 0 0;'>
                    Upload a Giemsa-stained thin blood smear image or connect a live microscope feed.
                </p>
            </div>
            <div style='background:#f8f8f8; border-radius:12px; padding:1.2rem;'>
                <div style='font-size:1.5rem;'>🧠</div>
                <strong>2. YOLO Inference</strong>
                <p style='font-size:0.85rem; color:#666; margin:0.3rem 0 0;'>
                    YOLOv8 scans the smear at cell level, detecting parasite morphology in milliseconds.
                </p>
            </div>
            <div style='background:#f8f8f8; border-radius:12px; padding:1.2rem;'>
                <div style='font-size:1.5rem;'>📊</div>
                <strong>3. Result & Confidence</strong>
                <p style='font-size:0.85rem; color:#666; margin:0.3rem 0 0;'>
                    A diagnosis (Positive/Negative) is returned with confidence score and bounding boxes.
                </p>
            </div>
            <div style='background:#f8f8f8; border-radius:12px; padding:1.2rem;'>
                <div style='font-size:1.5rem;'>📋</div>
                <strong>4. Clinical Review</strong>
                <p style='font-size:0.85rem; color:#666; margin:0.3rem 0 0;'>
                    Summary report generated for clinician review and patient referral if needed.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
 
 # ROUTER
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
 
 