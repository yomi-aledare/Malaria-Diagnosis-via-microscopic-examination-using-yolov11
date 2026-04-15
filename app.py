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
 
 