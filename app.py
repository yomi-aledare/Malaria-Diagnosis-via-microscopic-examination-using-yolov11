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
# Set the page title, icon, layout, and initial sidebar state for a polished look
st.set_page_config(
    page_title="BoboMal - Malaria Diagnosis App",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
# Custom CSS to style the entire app, including sidebar, buttons, and result banners
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
    min-width: 200px !important;
    max-width: 200px !important;
    width: 200px !important;
}
section[data-testid="stSidebar"] > div {
    padding: 0 0.6rem !important;
}
section[data-testid="stSidebar"] * { color: var(--cream) !important; }

/* ── Hide ALL sidebar collapse/expand controls — sidebar is always open ── */
/* Collapse button inside the open sidebar */
section[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"],
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button {
    display: none !important;
    visibility: hidden !important;
}
/* Re-open button shown when sidebar is collapsed */
button[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
    visibility: hidden !important;
}
/* Catch-all for any header/chevron buttons Streamlit injects */
section[data-testid="stSidebar"] header,
section[data-testid="stSidebar"] header button {
    display: none !important;
}

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

/* ── Sidebar nav buttons ── */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: var(--blue) !important;
    border: none !important;
    border-radius: 10px !important;
    text-align: left !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.65rem 1.1rem !important;
    margin-bottom: 0.2rem !important;
    transition: background 0.2s, color 0.2s !important;
    box-shadow: none !important;
    justify-content: flex-start !important;
    letter-spacing: 0.2px !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--blue) !important;
    color: white !important;
    transform: none !important;
    box-shadow: none !important;
}
/* Active page button — highlighted via st.session_state check + unique key CSS */
section[data-testid="stSidebar"] div:has(button[data-testid="baseButton-secondary"][key="nav_Home"]),
section[data-testid="stSidebar"] div:has(button[data-testid="baseButton-secondary"][key="nav_Diagnosis"]),
section[data-testid="stSidebar"] div:has(button[data-testid="baseButton-secondary"][key="nav_About"]) {
    border-radius: 10px;
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
    background: var(--blue);
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

/* ── Login card — targets the st.container via its data-testid key ── */
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stVerticalBlockBorderWrapper"]) 
    div[data-testid="stVerticalBlockBorderWrapper"] {
    background: white !important;
    border-radius: 20px !important;
    box-shadow: 0 20px 60px rgba(10,35,66,0.12) !important;
    border-top: 5px solid var(--blue) !important;
    padding: 2rem 2.5rem 2.5rem !important;
}

/* Fallback class for older Streamlit versions */
.login-card {
    background: white;
    border-radius: 20px;
    padding: 2.8rem 3rem;
    box-shadow: 0 20px 60px rgba(10,35,66,0.12);
    border-top: 5px solid var(--blue);
}

/* ── Buttons — blue bg, navy hover ── */
.stButton > button {
    background: var(--navy) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-family: 'Montserrat', sans-serif !important;
    letter-spacing: 0.5px !important;
    transition: background 0.2s, transform 0.1s !important;
}
.stButton > button:hover {
    background: var(--blue) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    background: var(--navy-lt) !important;
    transform: translateY(0) !important;
}

/* Secondary buttons (mode toggles) */
.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: var(--blue) !important;
    border: 2px solid var(--blue) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: var(--navy) !important;
    color: white !important;
    border-color: var(--navy) !important;
    transform: translateY(-1px) !important;
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
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0
if "annotated_image" not in st.session_state:
    st.session_state.annotated_image = None
if "parasite_count" not in st.session_state:
    st.session_state.parasite_count = 0
if "inference_time" not in st.session_state:
    st.session_state.inference_time = 0.0
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "parasite_name" not in st.session_state:
    st.session_state.parasite_name = None


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE - authentication logic and styled login form
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

        with st.container(border=True):
            st.markdown("""
            <style>
            div[data-testid="stVerticalBlockBorderWrapper"] {
                border-radius: 20px !important;
                border: none !important;
                background: white !important;
                box-shadow: 0 20px 60px rgba(10,35,66,0.12) !important;
                padding: 1.5rem 2rem 2rem !important;
                border-top: 5px solid #2986CC !important;
                outline: none !important;
            }
            </style>
            """, unsafe_allow_html=True)

            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", placeholder="Enter your password", type="password")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Login", width='stretch'):
                if username == "admin" and password == "admin123":
                    with st.spinner("Authenticating…"):
                        time.sleep(0.8)
                    st.session_state.authenticated = True
                    st.session_state.page = "Home"
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        st.markdown("""
        <p style='text-align:center; color:#aaa; font-size:0.76rem; margin-top:1.5rem;
                  font-family:"Roboto",sans-serif;'>
            🔐 Secure access · Medical use ONLY
        </p>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR  — logo, navigation, and logout button (only visible when authenticated)
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <div style="font-size:2.2rem;">🔬</div>
            <h1>BoboMal</h1>
            <p>Malaria <br> Diagnostic System</p>
        </div>
        """, unsafe_allow_html=True)

        # Inject active-state CSS for whichever page is current
        active = st.session_state.page
        st.markdown(f"""
        <style>
        div[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]
            div[data-testid="stVerticalBlock"] > div:nth-child(1) button {{
            background: {"var(--blue)" if active == "Home" else "transparent"} !important;
            color: {"white" if active == "Home" else "rgba(255,253,247,0.8)"} !important;
            box-shadow: {"0 4px 14px rgba(41,134,204,0.3)" if active == "Home" else "none"} !important;
        }}
        div[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]
            div[data-testid="stVerticalBlock"] > div:nth-child(2) button {{
            background: {"var(--blue)" if active == "Diagnosis" else "transparent"} !important;
            color: {"white" if active == "Diagnosis" else "rgba(255,253,247,0.8)"} !important;
            box-shadow: {"0 4px 14px rgba(41,134,204,0.3)" if active == "Diagnosis" else "none"} !important;
        }}
        div[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]
            div[data-testid="stVerticalBlock"] > div:nth-child(3) button {{
            background: {"var(--blue)" if active == "About" else "transparent"} !important;
            color: {"white" if active == "About" else "rgba(255,253,247,0.8)"} !important;
            box-shadow: {"0 4px 14px rgba(41,134,204,0.3)" if active == "About" else "none"} !important;
        }}
        </style>
        """, unsafe_allow_html=True)

        nav_items = [("🏠", "Home"), ("🔬", "Diagnosis"), ("ℹ️", "About")]
        for icon, label in nav_items:
            if st.button(f"{icon}  {label}", key=f"nav_{label}", width='stretch'):
                st.session_state.page = label
                st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div style='padding:0.6rem 0 1rem; font-size:0.73rem;
                    opacity:0.35; text-align:center; font-family:"Roboto",sans-serif;'>
            BoboMal v1.0
        </div>
        """, unsafe_allow_html=True)

        if st.button("Logout", width='stretch'):
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
# DIAGNOSIS PAGE  — image-only upload + full results panel
# ══════════════════════════════════════════════════════════════════════════════
def diagnosis_page():
    st.markdown("""
    <div class="page-header">
        <h2>Diagnosis</h2>
        <span class="badge">Analysis</span>
    </div>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([1.1, 1], gap="large")

    # ── Upload panel ──────────────────────────────────────────────────────────
    with left_col:
        st.markdown("""
        <div style='font-size:0.73rem; font-weight:700; letter-spacing:1.5px;
                    text-transform:uppercase; color:#999; margin-bottom:0.6rem;
                    font-family:"Montserrat",sans-serif;'>
            🖼  Image Feed
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload blood smear image",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            label_visibility="collapsed",
        )
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, width='stretch',
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

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶  Run Analysis", width='stretch')

        if run_btn:
            if not st.session_state.uploaded_file:
                st.warning("Please upload a blood smear image before running analysis.")
            else:
                with st.spinner("Running YOLOv11 inference…"):
                    (
                        st.session_state.diagnosis_result,
                        st.session_state.confidence,
                        st.session_state.annotated_image,
                        st.session_state.parasite_count,
                        st.session_state.inference_time,
                        st.session_state.parasite_name,
                    ) = run_yolo_inference(st.session_state.uploaded_file)
                st.session_state.show_summary = False

    # ── Results panel ─────────────────────────────────────────────────────────
    with right_col:
        st.markdown("""
        <div style='font-size:0.73rem; font-weight:700; letter-spacing:1.5px;
                    text-transform:uppercase; color:#999; margin-bottom:0.6rem;
                    font-family:"Montserrat",sans-serif;'>
            🔍 Analysis Result
        </div>
        """, unsafe_allow_html=True)

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

        # ── Annotated image ───────────────────────────────────────────────────
        if st.session_state.annotated_image is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style='font-size:0.73rem; font-weight:700; letter-spacing:1.5px;
                        text-transform:uppercase; color:#999; margin-bottom:0.4rem;
                        font-family:"Montserrat",sans-serif;'>
                🖼 Detection Output
            </div>
            """, unsafe_allow_html=True)
            st.image(
                st.session_state.annotated_image,
                width='stretch',
                caption="YOLOv11 bounding-box overlay",
            )

        if st.session_state.diagnosis_result:
            st.markdown("<br>", unsafe_allow_html=True)
            result_label = "Negative ✅" if st.session_state.diagnosis_result == "negative" else "Positive ⚠️"
            result_color = "#27ae60" if st.session_state.diagnosis_result == "negative" else "#e74c3c"
            score        = st.session_state.confidence
            n_parasites  = st.session_state.parasite_count
            duration     = st.session_state.inference_time
            parasite_name = st.session_state.parasite_name

            # Volume of infection via classify_parasitemia
            parasitemia_info = classify_parasitemia(parasites_per_ul=n_parasites * 500)
            vol_label = parasitemia_info["category"] if st.session_state.diagnosis_result == "positive" else "None"
            risk_label = parasitemia_info["risk"] if st.session_state.diagnosis_result == "positive" else "—"

            # Build optional Parasite Name row (only shown when positive)
            parasite_name_row = ""
            if st.session_state.diagnosis_result == "positive" and parasite_name:
                parasite_name_row = "P. " + parasite_name
            else:
                parasite_name_row = "N/A"

            st.markdown(f"""
            <div class="summary-card" style='margin-top:0.2rem;'>
                <h4>Analysis Summary</h4>
                <div class="stat-row">
                    <span>Category: </span>
                    <span class="val" style="color:{result_color};">{result_label}</span>
                </div>
                <div class="stat-row">
                    <span>Parasite Name</span>
                    <span class="val" style="color:{result_color};">{parasite_name_row}</span>
                </div>
                <div class="stat-row">
                    <span>Confidence Score</span>
                    <span class="val">{score*100:.1f}%</span>
                </div>
                <div class="stat-row">
                    <span>Total Count</span>
                    <span class="val">{n_parasites}</span>
                </div>
                <div class="stat-row">
                    <span>Volume of Infection</span>
                    <span class="val">{vol_label}</span>
                </div>
                <div class="stat-row">
                    <span>Risk Level</span>
                    <span class="val" style="color:{result_color};">{risk_label}</span>
                </div>
                <div class="stat-row">
                    <span>Inference Time</span>
                    <span class="val">{duration:.2f} s</span>
                </div>
                <div class="stat-row">
                    <span>Model</span>
                    <span class="val">YOLOv11</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABOUT PAGE  — split into separate st.markdown blocks so HTML renders correctly
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
            This Bsc project from <strong>Miva Open University</strong> aims to provide an accessible diagnostic tool for malaria, especially in resource-limited settings, by leveraging the power of deep learning and computer vision.
            It uses <strong>digital blood smear images</strong> to diagnose malaria parasites using <strong>YOLO (You Only Look Once) object detection</strong>. Giemsa-stained thin blood smears are analysed in real time, enabling fast,
            scalable, and accurate identification of <em>Plasmodium</em> species without requiring specialist laboratory personnel at the point of care.
        </p>
        <p>
            The YOLO model is trained on annotated microscopy images and can distinguish
            between infected and healthy red blood cells at the cell level, producing
            confidence scores and bounding-box overlays for clinical review.
        </p>

<p>
            Author: <strong>Yomi Aledare</strong><br>
            Matriculation Number: <strong>2023/C/DSC/161</strong><br>
            Supervisor: <strong>Chinonso Alaebo</strong> <br>
            Department:  <strong>Data Science</strong><br>
            School:  <strong>School of Computing</strong><br>
            Year: <em>2026</em><br>
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
# MODEL INFERENCE FUNCTION  —  loads YOLOv11 once and runs inference on uploaded images, returning structured results
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_yolo_model():
    """Load the YOLOv11 model once and cache it across Streamlit reruns."""
    weight_path = Path("weights") / "best.pt"
    if not weight_path.exists():
        st.error(f"Model weights not found at '{weight_path}'. "
                 "Please ensure 'weights/best.pt' exists.")
        st.stop()
    return YOLO(str(weight_path))

def run_yolo_inference(image):
    """
    Run YOLOv11 inference on a single image.

    Parameters
    ----------
    image : UploadedFile or PIL.Image.Image

    Returns
    -------
    diagnosis      : str        — "positive" or "negative"
    confidence     : float      — top detection confidence (0–1)
    annotated_img  : PIL.Image.Image — image with bounding boxes drawn
    parasite_count : int        — positive-class count if positive, total count if negative
    inference_time : float      — wall-clock seconds for model.predict()
    parasite_name  : str | None — "Falciparum", "Vivax", or None
    """
    import numpy as np

    CONF_THRESHOLD = 0.25

    # Classes that map to each Plasmodium species
    FALCIPARUM_CLASSES = {'Seg-F', 'F-R', 'F-S', 'F-T'}
    VIVAX_CLASSES      = {'V-G', 'V-R', 'V-S', 'V-T'}
    POSITIVE_CLASSES   = FALCIPARUM_CLASSES | VIVAX_CLASSES

    model = load_yolo_model()

    # Accept either a Streamlit UploadedFile or a PIL Image
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    pil_img = image.convert("RGB")

    # ── Resize to 640×640 before inference ───────────────────────────────────
    pil_img = pil_img.resize((640, 640), Image.LANCZOS)
    img_array = np.array(pil_img)

    t0 = time.perf_counter()
    results = model.predict(source=img_array, conf=CONF_THRESHOLD, verbose=False)
    inference_time = time.perf_counter() - t0

    # ── Collect detections ────────────────────────────────────────────────────
    all_confidences      = []
    positive_confidences = []
    diagnosis            = "negative"
    parasite_name        = None
    detected_species     = set()   # track which positive species appear

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        confs   = r.boxes.conf.cpu().tolist()
        classes = r.boxes.cls.cpu().tolist()
        all_confidences.extend(confs)

        for conf, cls_idx in zip(confs, classes):
            class_name = model.names[int(cls_idx)]
            if class_name in FALCIPARUM_CLASSES:
                diagnosis = "positive"
                detected_species.add("Falciparum")
                positive_confidences.append(conf)
            elif class_name in VIVAX_CLASSES:
                diagnosis = "positive"
                detected_species.add("Vivax")
                positive_confidences.append(conf)

    # Determine parasite name (prefer Falciparum if both detected)
    if "Falciparum" in detected_species and "Vivax" in detected_species:
        parasite_name = "Falciparum & Vivax"
    elif "Falciparum" in detected_species:
        parasite_name = "Falciparum"
    elif "Vivax" in detected_species:
        parasite_name = "Vivax"

    # Count: positive-class detections when positive, all detections when negative
    parasite_count = len(positive_confidences) if diagnosis == "positive" else len(all_confidences)
    top_confidence = float(max(positive_confidences if diagnosis == "positive" else all_confidences)) \
                     if (positive_confidences or all_confidences) else 0.0

    # ── Draw bounding boxes onto the image ───────────────────────────────────
    annotated_bgr = results[0].plot()          # numpy BGR array with boxes drawn
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)

    return diagnosis, top_confidence, annotated_pil, parasite_count, inference_time, parasite_name

def classify_parasitemia(parasitemia_percent=None, parasites_per_ul=None):
    # If percentage is provided, convert to parasites/µL for classification
    if parasitemia_percent is not None:
        # Approximate conversion: 1% parasitemia ≈ 50,000 parasites/µL
        # (assuming normal RBC count of 5 million/µL)
        parasites_per_ul_est = parasitemia_percent * 50000
    else:
        parasites_per_ul_est = parasites_per_ul
    
    # Classification logic based on parasites/µL
    if parasites_per_ul_est is None:
        raise ValueError("Either parasitemia_percent or parasites_per_ul must be provided")
    
    if parasites_per_ul_est < 50:
        category = "Submicroscopic / low"
        risk = "Very low"
        percent_range = "<0.001%"
        parasite_range = "<50/µL"
    elif 50 <= parasites_per_ul_est < 5000:
        category = "Low parasitemia"
        risk = "Low"
        percent_range = "0.002–0.1%"
        parasite_range = "50–5,000/µL"
    elif 5000 <= parasites_per_ul_est < 100000:
        category = "Moderate"
        risk = "Moderate"
        percent_range = "0.1–2%"
        parasite_range = "5,000–100,000/µL"
    elif 100000 <= parasites_per_ul_est < 250000:
        category = "High"
        risk = "High"
        percent_range = "2–5%"
        parasite_range = "100,000–250,000/µL"
    else:  # >= 250000
        category = "Severe malaria risk (especially P. falciparum)"
        risk = "Severe"
        percent_range = ">5–10%"
        parasite_range = ">250,000–500,000/µL"
    
    # Prepare result
    result = {
        'category': category,
        'risk': risk,
        'percent_range': percent_range,
        'parasite_range': parasite_range,
        'input_percent': parasitemia_percent,
        'input_parasites_ul': parasites_per_ul,
        'estimated_parasites_ul': round(parasites_per_ul_est, 2) if parasitemia_percent is not None else parasites_per_ul
    }
    
    return result
# ══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTER - renders the appropriate page based on authentication and sidebar selection
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