import streamlit as st
import torch
import pickle
import pandas as pd

from sasrec_model import SASRecWithLLMAndIPS
from recommender import recommend_movies

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="MovieVerse AI",
    page_icon="🎬",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    with open("data/movies.pkl", "rb") as f:
        movies = pickle.load(f)

    n_items = movies["item_id"].nunique()

    model = SASRecWithLLMAndIPS(n_items=n_items)
    model.load_state_dict(
        torch.load("models/model_llm_ips.pth", map_location="cpu")
    )
    model.eval()
    return model

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    with open("models/item_embeddings.pkl", "rb") as f:
        item_embeddings = pickle.load(f)

    with open("data/movies.pkl", "rb") as f:
        movies = pickle.load(f)

    return item_embeddings, movies

model = load_model()
item_embeddings, movies = load_data()

# =========================
# UI
# =========================
st.title("🎬 MovieVerse AI")
st.caption("SASRec + LLM + Inverse Propensity Scoring")

watched_titles = st.multiselect(
    "Pilih film yang pernah ditonton",
    movies["title"].tolist()
)

generate = st.button("✨ Tampilkan Rekomendasi")

if generate:
    if len(watched_titles) < 3:
        st.warning("⚠️ Pilih minimal 3 film")
    else:
        item_map = dict(zip(movies["title"], movies["item_id"]))
        user_sequence = [item_map[t] for t in watched_titles]

        with st.spinner("🔍 Menghitung rekomendasi..."):
            recommendations = recommend_movies(
                model,
                user_sequence,
                item_embeddings,
                movies,
                k=10
            )

        st.subheader("✨ 10 Rekomendasi Film Untuk Anda")

        cols = st.columns(5)
        for i, rec in enumerate(recommendations):
            with cols[i % 5]:
                st.markdown(f"""
                <div style="
                    background:white;
                    padding:16px;
                    border-radius:14px;
                    box-shadow:0 6px 18px rgba(0,0,0,.1);
                    min-height:120px;
                ">
                    <h4 style="font-size:14px;">{rec['title']}</h4>
                    <p style="font-size:12px;color:#64748b;">
                        {rec['genres']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

st.markdown("""
<hr>
<center style="color:#64748b;font-size:13px;">
MovieVerse AI • SASRec + LLM + IPS
</center>
""", unsafe_allow_html=True)
