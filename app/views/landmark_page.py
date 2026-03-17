"""
Landmark Extractor Page

Extract and process MediaPipe landmarks from keyframe images using
the RQ pipeline (ported from asl_landmark_extractor CLI tool).
"""

import streamlit as st


def render():
    """Render the landmark extractor page."""
    st.caption("Extract and process MediaPipe landmarks from keyframe images.")
    st.info("🚧 This page is being rebuilt with the new RQ landmark extraction pipeline.")
