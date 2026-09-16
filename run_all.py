"""
Run-All script for the Retail AI Co-Pilot.

Usage:
    python run_all.py

Launches the Streamlit app, which opens automatically in your browser at
http://localhost:8501. From there, create an account and upload your own
sales data -- there's no pre-built demo dataset to regenerate; everything
starts from your own upload.
"""
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(PROJECT_ROOT, "app", "streamlit_app.py")


def main():
    print("=" * 60)
    print("Launching the Retail AI Co-Pilot...")
    print("It will open automatically in your browser at:")
    print("    http://localhost:8501")
    print("Press Ctrl+C in this window to stop it.")
    print("=" * 60 + "\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", APP_PATH])


if __name__ == "__main__":
    main()
