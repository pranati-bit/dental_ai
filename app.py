from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import backend.main as model

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
st.set_page_config(page_title="Dental AI Universe", page_icon="🦷", layout="wide")

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: radial-gradient(ellipse at 70% 0%, #111b35 0%, #070a12 48%, #03050a 100%); color:#edf5ff; }
.block-container { padding-top: 1.6rem; max-width: 1500px; }
h1,h2,h3 { letter-spacing:-.035em; }
.neon { color:#6cecff; text-shadow:0 0 18px #19bde866; }
.panel { background:linear-gradient(145deg,rgba(18,29,52,.88),rgba(7,11,22,.94)); border:1px solid #24415d; border-radius:18px; padding:22px; }
.smallmuted { color:#98a9c2; }
div[data-testid="stMetric"] { background:#0c1424; border:1px solid #203750; padding:16px; border-radius:14px; }
.stButton>button { border:1px solid #39d9ee; background:#102b42; color:#eafcff; border-radius:12px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="neon">🦷 SEE BEYOND THE SMILE.</h1>', unsafe_allow_html=True)
st.markdown("Artificial Intelligence. Dental Intelligence. Explainable Decisions.")
st.caption("Research prototype · Not a clinical diagnostic or treatment tool")

sections = ["01 · 3D Digital Universe UI", "02 · AI Scanner & Live Inference",
            "03 · Model Performance Lab", "04 · Explainable AI Gallery"]
section = st.sidebar.radio("NAVIGATION CONSOLE", sections)

if section == sections[0]:
    st.markdown("## Enter the Lab")
    st.markdown('<div class="panel"><h2 class="neon">THE DIGITAL DENTAL LABORATORY</h2><p class="smallmuted">Explore the intersection of dental imaging, machine learning and explainable AI.</p></div>', unsafe_allow_html=True)
    st.markdown("### Laboratory systems")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="panel"><h3>◉ AI Scanner</h3><p>Run the saved braces and crowding models on a new image.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel"><h3>⌁ Model Lab</h3><p>Inspect recorded evaluation metrics and confusion matrices.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="panel"><h3>✧ Explainability</h3><p>Explore existing patient explanations and new-upload sensitivity maps.</p></div>', unsafe_allow_html=True)
    st.info("This Streamlit edition uses a cinematic dark interface and interactive navigation. Streamlit is not a native React Three Fiber canvas, so the 3D portfolio-style scene is represented as a lightweight lab landing page.")

elif section == sections[1]:
    st.header("AI Scanner & Live Inference")
    st.write("Upload a JPG, PNG or WebP image. Both saved classifiers will run independently.")
    upload = st.file_uploader("Upload dental image", type=["jpg","jpeg","png","webp"])
    if upload:
        try:
            im = Image.open(upload).convert("RGB")
            st.image(im, caption="Uploaded dental image", use_container_width=True)
            with st.spinner("Running saved models and generating a new occlusion-sensitivity map..."):
                arr, z, box = model.preprocess(im)
                bp = model.dense_predict(z, model.BRACES_LAYERS)
                bi = int(np.argmax(bp))
                heat, cp = model.make_occlusion(arr)
                overlay = Image.open(__import__("io").BytesIO(__import__("base64").b64decode(model.overlay_data_url(arr, heat, 0.62).split(",",1)[1])))
            st.success("Inference completed using the saved project weights.")
            a,b = st.columns(2)
            with a:
                st.subheader("Braces requirement model")
                st.metric("Predicted class", model.CLASSES[bi])
                st.write({k: round(float(v),4) for k,v in zip(model.CLASSES,bp)})
                st.caption("Three-class research prediction: no_need / borderline / braces_likely.")
            with b:
                st.subheader("Crowding model")
                st.metric("Predicted class", "crowding" if cp >= .5 else "no_crowding")
                st.metric("Model score", f"{cp:.1%}", help="Probability assigned to the crowding class; not calibrated clinical risk.")
            st.subheader("New-upload explainability")
            opacity = st.slider("Heatmap overlay opacity",0.0,1.0,0.62,0.02)
            overlay_url = model.overlay_data_url(arr, heat, opacity)
            import base64, io
            st.image(Image.open(io.BytesIO(base64.b64decode(overlay_url.split(",",1)[1]))), caption="Occlusion sensitivity overlay generated for this upload", use_container_width=True)
            st.caption("Occlusion sensitivity: image patches are neutralized and changes in crowding probability are mapped. This is model sensitivity, not clinical evidence.")
            st.warning("Research prototype only. These predictions are not clinically validated and must not be used to make treatment decisions.")
        except Exception as e:
            st.error(f"Inference failed: {e}")

elif section == sections[2]:
    st.header("Model Performance Lab")
    metrics_path = RESULTS / "crowding_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.subheader("Crowding classifier — saved evaluation")
        st.json(metrics)
    else:
        st.warning("crowding_metrics.json was not found.")
    braces_eval = model.BRACES.get("eval", {})
    st.subheader("Braces classifier — saved evaluation")
    st.json(braces_eval)
    for title, filename in [("Crowding confusion matrix","crowding_confusion.png"),("Exported model confusion matrix","confusion_matrix.png"),("Training loss","crowding_loss.png")]:
        p = RESULTS / filename
        if p.exists():
            st.subheader(title)
            st.image(str(p), use_container_width=True)
    st.caption("Only metrics and plots present in the supplied project are shown. No values are fabricated.")

else:
    st.header("Explainable AI Gallery")
    exp = RESULTS / "explain"
    idx_path = RESULTS / "index.csv"
    score_path = exp / "explained_patients.csv"
    if not idx_path.exists():
        st.warning("Patient image index is unavailable.")
    else:
        idx = pd.read_csv(idx_path, dtype={"patient_id":str})
        scores = pd.read_csv(score_path, dtype={"patient":str}) if score_path.exists() else pd.DataFrame()
        files = sorted(exp.glob("*_crowding.png"))
        choices = []
        for p in files:
            pid = p.name.split("_")[0]
            row = idx[idx.patient_id == pid]
            if row.empty: continue
            photo = ROOT / str(row.iloc[0]["path"])
            if photo.exists(): choices.append((pid,p,photo))
        if not choices:
            st.info("No correctly matched patient image/explanation pairs were found.")
        else:
            labels = [f"{pid} — {p.name}" for pid,p,_ in choices]
            selected = st.selectbox("Select patient explanation", labels)
            pid,p,photo = choices[labels.index(selected)]
            left,right = st.columns(2)
            with left:
                st.subheader("Original patient image")
                st.image(str(photo), use_container_width=True)
            with right:
                st.subheader("Existing XAI explanation")
                st.image(str(p), use_container_width=True)
            if not scores.empty and "patient" in scores.columns:
                score = scores[scores.patient.astype(str)==pid]
                if not score.empty and "p_crowding" in score.columns:
                    prob=float(score.iloc[0].p_crowding)
                    st.metric("Recorded crowding score",f"{prob:.1%}")
                    st.caption("Score shown only when available in the matched explanation metadata.")
            st.caption("Gallery items are pre-existing explanations matched by patient key. They are not reused for new uploads.")
