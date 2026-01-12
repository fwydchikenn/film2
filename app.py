import streamlit as st
import pickle
import torch
import pandas as pd
import numpy as np
import os

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
# PATH HANDLING (STREAMLIT SAFE)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "movies.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_llm_ips.pth")
EMB_PATH = os.path.join(BASE_DIR, "models", "item_embeddings.pkl")

DEVICE = torch.device("cpu")

# =========================
# LOAD DATA & MODEL
# =========================
@st.cache_resource
def load_all():
    # Load movies
    with open(DATA_PATH, "rb") as f:
        movies = pickle.load(f)

    # Load item embeddings
    with open(EMB_PATH, "rb") as f:
        item_embeddings = pickle.load(f)

    # Model params (HARUS SAMA DENGAN TRAINING)
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

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model.eval()

    return movies, item_embeddings, model

movies, item_embeddings, model = load_all()

# =========================
# HEADER
# =========================
st.markdown("""
<div style="text-align:center; padding:20px;">
    <h1>🎬 MovieVerse AI</h1>
    <p style="color:#64748b;">
        SASRec + LLM + Inverse Propensity Scoring
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR – USER HISTORY
# =========================
with st.sidebar:
    st.header("👤 Histori Tontonan")

    watched_titles = st.multiselect(
        "Pilih minimal 5 film yang pernah ditonton",
        movies["title"].values
    )

    st.markdown("---")
    run_button = st.button("🚀 Tampilkan Rekomendasi")

# =========================
# MAIN LOGIC
# =========================
MIN_WATCH = 5
TOP_K = 10

if not run_button:
    st.info("👈 Pilih film di sidebar lalu tekan **Tampilkan Rekomendasi**")

elif len(watched_titles) < MIN_WATCH:
    st.warning(f"⚠️ Minimal pilih **{MIN_WATCH} film**")

else:
    # Convert titles → item_id sequence
    user_sequence = (
        movies[movies["title"].isin(watched_titles)]
        .sort_values("item_id")["item_id"]
        .tolist()
    )

    with st.spinner("🔍 Menganalisis preferensi Anda..."):
        recommendations = recommend_movies(
            model=model,
            user_sequence=user_sequence,
            item_embeddings=item_embeddings,
            movies_df=movies,
            k=TOP_K,
            use_llm=True
        )

    st.subheader("✨ 10 Rekomendasi Film Untuk Anda")

    # ===== GRID 5 x 2 =====
    cols = st.columns(5)

    for i, rec in enumerate(recommendations):
        with cols[i % 5]:
            st.markdown(f"""
            <div style="
                background:#ffffff;
                padding:18px;
                border-radius:14px;
                text-align:center;
                box-shadow:0 8px 20px rgba(15,23,42,0.08);
                margin-bottom:20px;
            ">
                <h4 style="font-size:14px; min-height:48px;">
                    {rec['title']}
                </h4>
                <p style="font-size:12px; color:#64748b;">
                    🎭 {rec['genres']}
                </p>
                <p style="font-size:12px; color:#2563eb; font-weight:600;">
                    Score: {rec['score']:.4f}
                </p>
            </div>
            """, unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<hr>
<div style="text-align:center; color:#64748b; font-size:13px;">
MovieVerse AI • SASRec + LLM + IPS<br>
Thesis-Grade Recommendation System
</div>
""", unsafe_allow_html=True)
