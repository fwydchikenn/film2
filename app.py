import streamlit as st
import pickle
import torch
import os
import pandas as pd
import numpy as np

from sasrec_model import SASRecWithLLMAndIPS
from recommender import recommend_movies

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="🎬 MovieVerse AI",
    page_icon="🎬",
    layout="wide"
)

# =========================
# BASE DIRECTORY
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "movies.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_llm_ips.pth")
EMB_PATH = os.path.join(BASE_DIR, "models", "item_embeddings.pkl")

DEVICE = torch.device("cpu")

# =========================
# DEBUG PANEL (WAJIB ADA SEKARANG)
# =========================
with st.expander("🔍 Debug File System"):
    st.write("📂 BASE_DIR:", BASE_DIR)

    if os.path.exists(BASE_DIR):
        st.write("📁 Root files:", os.listdir(BASE_DIR))

    if os.path.exists(os.path.join(BASE_DIR, "data")):
        st.write("📁 data/:", os.listdir(os.path.join(BASE_DIR, "data")))
    else:
        st.error("❌ Folder `data/` TIDAK ADA")

    if os.path.exists(os.path.join(BASE_DIR, "models")):
        st.write("📁 models/:", os.listdir(os.path.join(BASE_DIR, "models")))
    else:
        st.error("❌ Folder `models/` TIDAK ADA")

# =========================
# LOAD ALL (SAFE)
# =========================
@st.cache_resource
def load_all():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Tidak ditemukan: {DATA_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Tidak ditemukan: {MODEL_PATH}")

    if not os.path.exists(EMB_PATH):
        raise FileNotFoundError(f"❌ Tidak ditemukan: {EMB_PATH}")

    # Load data
    with open(DATA_PATH, "rb") as f:
        movies = pickle.load(f)

    with open(EMB_PATH, "rb") as f:
        item_embeddings = pickle.load(f)

    n_items = movies["item_id"].nunique()

    model = SASRecWithLLMAndIPS(
        n_items=n_items,
        embedding_dim=128,
        n_heads=2,
        n_layers=2,
        max_seq_len=50,
        dropout=0.2,
        llm_dim=768
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    return movies, item_embeddings, model

# =========================
# LOAD
# =========================
movies, item_embeddings, model = load_all()

# =========================
# HEADER
# =========================
st.markdown("""
<div style="text-align:center;padding:20px;">
<h1>🎬 MovieVerse AI</h1>
<p style="color:#64748b;">SASRec + LLM + IPS</p>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("🎥 Histori Tontonan")

    watched_titles = st.multiselect(
        "Pilih minimal 5 film",
        movies["title"].values
    )

    run_btn = st.button("🚀 Tampilkan Rekomendasi")

# =========================
# MAIN
# =========================
MIN_WATCH = 5

if not run_btn:
    st.info("👈 Pilih film lalu klik tombol rekomendasi")

elif len(watched_titles) < MIN_WATCH:
    st.warning(f"⚠️ Minimal pilih {MIN_WATCH} film")

else:
    user_sequence = (
        movies[movies["title"].isin(watched_titles)]
        .sort_values("item_id")["item_id"]
        .tolist()
    )

    with st.spinner("🔍 Menghasilkan rekomendasi..."):
        recs = recommend_movies(
            model=model,
            user_sequence=user_sequence,
            item_embeddings=item_embeddings,
            movies_df=movies,
            k=10,
            use_llm=True
        )

    st.subheader("✨ 10 Rekomendasi Film Untuk Anda")

    cols = st.columns(5)
    for i, rec in enumerate(recs):
        with cols[i % 5]:
            st.markdown(f"""
            <div style="background:#fff;padding:16px;border-radius:14px;
            box-shadow:0 8px 20px rgba(0,0,0,0.08);margin-bottom:20px;text-align:center">
                <h4 style="font-size:14px;min-height:48px">{rec['title']}</h4>
                <p style="font-size:12px;color:#64748b">🎭 {rec['genres']}</p>
                <p style="font-size:12px;font-weight:600;color:#2563eb">
                    Score: {rec['score']:.4f}
                </p>
            </div>
            """, unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<hr>
<div style="text-align:center;color:#64748b;font-size:13px">
MovieVerse AI • Thesis-Grade Recommender System
</div>
""", unsafe_allow_html=True)
