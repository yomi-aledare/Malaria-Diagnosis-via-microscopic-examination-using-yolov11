#importing required libraries
import cv2
from pathlib import Path
import os
import sys
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Set page configuration
st.set_page_config(page_title="Malaria Diagnosis App", page_icon=":camera:", layout="wide")

#header of the app
st.header("Malaria Diagnosis App")
