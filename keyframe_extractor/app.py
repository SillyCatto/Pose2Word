"""
WLASL Dataset Preparation Tool — Main Entry Point

A Streamlit app for preparing ASL (American Sign Language) datasets:
  1. Keyframe Extractor: Extract keyframes from sign language videos
  2. Landmark Extractor: Convert keyframes to MediaPipe landmarks

Business logic is delegated to separate modules:
  - video_utils.py        → Video I/O
  - algorithms.py         → Keyframe extraction algorithms
  - file_utils.py         → Frame saving
  - landmark_extractor.py → MediaPipe landmark extraction
  - data_exporter.py      → Numpy export
"""

import streamlit as st

from views import keyframe_page, landmark_page, flow_page, training_page, manual_keyframe_page


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="WLASL Dataset Preparation Tool",
    page_icon="🖐️",
    layout="wide",
)


# =============================================================================
# CUSTOM STYLES
# =============================================================================
st.markdown(
    """
    <style>
    /* Make tabs larger */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 10px 24px;
        font-size: 18px;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# MAIN APP
# =============================================================================
st.title("🖐️ WLASL Dataset Preparation Tool")

# Create tabs
tab_keyframe, tab_landmark, tab_flow, tab_training, tab_manual = st.tabs([
    "📹 Keyframe Extractor", 
    "🦴 Landmark Extractor",
    "🌊 Flow Extractor",
    "🤖 Model Training",
    "🖱️ Manual Keyframe Selection",
])

# Render pages within tabs
with tab_keyframe:
    keyframe_page.render()

with tab_landmark:
    landmark_page.render()

with tab_flow:
    flow_page.render()

with tab_training:
    training_page.render()

with tab_manual:
    manual_keyframe_page.render()
