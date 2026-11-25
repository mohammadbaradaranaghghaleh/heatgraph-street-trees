"""Streamlit Community Cloud entrypoint (main file path: dashboard/streamlit_app.py).

It runs the repository's app.py unchanged. Keeping the entrypoint in this folder makes Community Cloud
install dashboard/requirements.txt (the dashboard's few light dependencies) instead of the full
pipeline requirements at the repository root (PyTorch, geospatial stack).
Locally, `streamlit run app.py` from the repository root is equivalent.
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "app.py"), run_name="__main__")
