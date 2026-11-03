# Dental AI Universe — Streamlit edition

## Run on Windows

1. Extract this folder and open `braces_proj_code` in VS Code.
2. Open a terminal in the project root (the folder containing `app.py`).
3. Run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r backend\requirements.txt
streamlit run app.py
```

4. Open the local URL Streamlit prints (normally http://localhost:8501).

If PowerShell blocks activation, run this once in that terminal:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
Then activate the environment again.

This edition uses Streamlit for the full application. It preserves the saved braces and crowding model weights, reads the real evaluation files, and generates a new occlusion-sensitivity explanation for each uploaded image. The 3D landing page is a Streamlit-styled lab entry page, not a native React Three Fiber scene.

Research prototype only; not clinically validated or intended for treatment decisions. De-identify patient images and verify consent before sharing or deploying.
