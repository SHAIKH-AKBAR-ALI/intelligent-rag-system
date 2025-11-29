# Streamlit Cloud entry point
# This file should be in the root directory for Streamlit Cloud deployment

import sys
import os
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(current_dir))

# Import and run the main app
try:
    from src.ui.streamlit_app import *
except ImportError:
    # Fallback: try to import directly
    import streamlit as st
    st.error("Failed to import the main application. Please check the file structure.")
    st.info("Expected structure: src/ui/streamlit_app.py")
    st.stop()