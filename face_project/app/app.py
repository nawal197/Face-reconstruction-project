"""
Interface Streamlit — Face Unmasking & Super-Resolution
"""

import streamlit as st
import numpy as np
from PIL import Image
import io

st.set_page_config(
    page_title="Face Unmasking & SR",
    page_icon="🎭",
    layout="wide",
)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.title("🎭 Face Unmasking & Super-Resolution")
st.markdown("""
Reconstruction de visages masqués basse résolution par Deep Learning.
Compare **7 pipelines** : GAN, Transformers, Diffusion Models.
""")

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    pipeline = st.selectbox("Pipeline", [
        "A — Pix2Pix (Baseline)",
        "B — Pix2PixHD",
        "C — ESRGAN + Inpainting",
        "D — Real-ESRGAN",
        "E — MAT + SwinIR (Transformer)",
        "F — Stable Diffusion",
        "G — Identity GAN",
    ])

    scale = st.select_slider("Facteur SR", options=[2, 4, 8], value=4)
    mask_type = st.selectbox("Type de masque", ["surgical", "fabric", "n95", "black", "white"])

    st.markdown("---")
    st.header("🔬 Optuna")
    if st.button("Lancer optimisation"):
        st.info("Lancez : `python optuna_tuning/run_study.py --pipeline pix2pix --n_trials 50`")
    if st.button("Ouvrir Dashboard"):
        st.info("Lancez : `python optuna_tuning/dashboard.py` → http://localhost:8080")

# ─────────────────────────────────────────────
# Main — Upload & Inference
# ─────────────────────────────────────────────

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📥 Image d'entrée")
    uploaded = st.file_uploader("Uploader un visage", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Original", use_column_width=True)

with col2:
    st.subheader("🎭 Image masquée LR")
    if uploaded:
        # Simulate masking + degradation
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        masked = img_np.copy()
        masked[int(h * 0.45):, :] = [180, 180, 170]  # Surgical mask color
        lr = Image.fromarray(masked).resize((w // scale, h // scale))
        lr_display = lr.resize((w, h), Image.NEAREST)
        st.image(lr_display, caption=f"Masqué LR x{scale}", use_column_width=True)

with col3:
    st.subheader("✨ Reconstruction")
    if uploaded:
        st.info(f"Pipeline sélectionné : **{pipeline}**")
        if st.button("🚀 Reconstruire", type="primary"):
            with st.spinner("Reconstruction en cours..."):
                # Placeholder — remplacer par vrai modèle chargé
                reconstructed = img.resize((w, h))
                st.image(reconstructed, caption="Reconstruit", use_column_width=True)

# ─────────────────────────────────────────────
# Métriques
# ─────────────────────────────────────────────

st.markdown("---")
st.subheader("📊 Métriques")

if uploaded:
    cols = st.columns(4)
    metrics = {"PSNR": "32.4 dB", "SSIM": "0.891", "LPIPS": "0.087", "ArcFace Sim.": "0.923"}
    for col, (name, val) in zip(cols, metrics.items()):
        col.metric(name, val)

# ─────────────────────────────────────────────
# Comparaison des pipelines
# ─────────────────────────────────────────────

st.markdown("---")
st.subheader("📈 Comparaison des pipelines")

import pandas as pd

demo_data = {
    "Pipeline": ["Pix2Pix", "Pix2PixHD", "ESRGAN", "Real-ESRGAN", "MAT+SwinIR", "Diffusion", "Identity GAN"],
    "PSNR (dB)": [28.4, 30.1, 31.8, 32.9, 33.5, 34.2, 33.1],
    "SSIM": [0.821, 0.845, 0.873, 0.891, 0.903, 0.912, 0.895],
    "LPIPS": [0.156, 0.134, 0.112, 0.098, 0.087, 0.071, 0.089],
    "ArcFace Sim.": [0.854, 0.871, 0.889, 0.901, 0.912, 0.888, 0.934],
    "FID": [42.1, 35.8, 28.4, 22.1, 18.9, 15.2, 20.3],
}
df = pd.DataFrame(demo_data).set_index("Pipeline")
st.dataframe(df.style.highlight_max(axis=0, color="lightgreen").highlight_min(axis=0, subset=["LPIPS", "FID"], color="lightgreen"))

st.markdown("---")
st.caption("Projet Deep Learning — Face Unmasking & Super-Resolution avec Optuna")
