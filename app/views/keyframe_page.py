"""
Keyframe Extractor Page

Extract keyframes from sign language videos using Farneback + MediaPipe
hybrid pipeline (ported from asl_keyframe_extractor CLI tool).
"""

import streamlit as st


def render():
    """Render the keyframe extractor page."""
    st.caption("Extract keyframes from sign language videos for dataset preparation.")
    st.info("🚧 This page is being rebuilt with the new hybrid keyframe extraction pipeline.")
