"""Root entry point for Streamlit Community Cloud deployment."""

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parent / "dashboard" / "app.py"),
    run_name="__main__",
)
