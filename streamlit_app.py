# Streamlit Cloud entry point
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now import the main app
exec(open('src/ui/streamlit_app.py').read())