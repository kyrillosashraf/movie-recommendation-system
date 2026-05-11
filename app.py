"""
app.py  –  Movie Recommendation System  (no Surprise dependency)
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎬", layout="wide")

st.markdown("""
<style>
    .rec-card {
        background: linear-gradient(135deg,#1e2130,#252a40);
        border-radius:12px; padding:16px 20px; margin-bottom:10px;
        border-left:4px solid #e50914;
    }
    .rec-title { font-size:1.05rem; font-weight:700; color:#f0f0f0; }
    .rec-genre { font-size:0.8rem; color:#aaa; margin-top:4px; }
    .score-bar { height:6px; border-radius:3px; background:#e50914; margin-top:8px; }
    .metric-box { background:#1e2130; border-radius:10px; padding:18px; text-align:center; }
    .metric-val { font-size:2rem; font-weight:800; color:#e50914; }
    .metric-lbl { font-size:0.85rem; color:#aaa; }
</style>
""", unsafe_allow_html=True)

#── Load artefacts ─────────────────────────────────────────────────────────────

REQUIRED = ["svd_artefacts.pkl", "cosine_sim.pkl", "cb_data.pkl", "movies.pkl", "ratings.pkl"]

@st.cache_resource(show_spinner="Loading models...")
def load_models():
    base = Path("models")
    with open(base / "svd_artefacts.pkl", "rb") as f:
        svd = pickle.load(f)          # dict with R_predicted, user2idx, movie2idx …
    with open(base / "cosine_sim.pkl", "rb") as f:
        cos_sim = pickle.load(f)
    cb_data  = pd.read_pickle(base / "cb_data.pkl")
    movies   = pd.read_pickle(base / "movies.pkl")
    ratings  = pd.read_pickle(base / "ratings.pkl")
    return svd, cos_sim, cb_data, movies, ratings
base = Path("models")

models_ready = all((base / fn).exists() for fn in REQUIRED)
# ── Prediction helper ──────────────────────────────────────────────────────────
def predict_rating(user_id, movie_id, svd):
    u2i = svd["user2idx"]
    m2i = svd["movie2idx"]
    if user_id not in u2i or movie_id not in m2i:
        return svd["global_mean"]
    return float(svd["R_predicted"][u2i[user_id], m2i[movie_id]])

# ── Recommendation functions ───────────────────────────────────────────────────
def cb_recommend(title, cb_data, cosine_sim, n=10):
    indices = pd.Series(cb_data.index, index=cb_data["title"])
    if title not in indices:
        return pd.DataFrame()
    idx    = indices[title]
    scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:n+1]
    result = cb_data[["movieId", "title", "genres"]].iloc[[s[0] for s in scores]].copy()
    result["score"] = [s[1] for s in scores]
    return result.reset_index(drop=True)


def cf_recommend(user_id, svd, movies_df, ratings_df, n=10):
    rated_ids  = set(ratings_df[ratings_df["userId"] == user_id]["movieId"])
    unseen_ids = list(set(movies_df["movieId"]) - rated_ids)
    preds      = sorted(
        [(mid, predict_rating(user_id, mid, svd)) for mid in unseen_ids],
        key=lambda x: x[1], reverse=True
    )
    top_ids, top_scores = zip(*preds[:n])
    result = movies_df[movies_df["movieId"].isin(top_ids)][["movieId", "title", "genres"]].copy()
    result["score"] = result["movieId"].map(dict(zip(top_ids, top_scores)))
    return result.sort_values("score", ascending=False).reset_index(drop=True)


def hybrid_recommend(user_id, title, svd, movies_df, ratings_df, cb_data, cosine_sim,
                     n=10, cb_weight=0.4, cf_weight=0.6):
    # CB scores
    indices = pd.Series(cb_data.index, index=cb_data["title"])
    if title in indices:
        idx      = indices[title]
        sim_raw  = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:]
        cb_ids   = cb_data.iloc[[s[0] for s in sim_raw]]["movieId"].values
        cb_vals  = np.array([s[1] for s in sim_raw])
        cb_norm  = cb_vals / (cb_vals.max() + 1e-9)
        cb_df    = pd.DataFrame({"movieId": cb_ids, "cb_score": cb_norm})
    else:
        cb_df = pd.DataFrame(columns=["movieId", "cb_score"])

    # CF scores
    rated_ids  = set(ratings_df[ratings_df["userId"] == user_id]["movieId"])
    unseen_ids = list(set(movies_df["movieId"]) - rated_ids)
    cf_raw     = [(mid, predict_rating(user_id, mid, svd)) for mid in unseen_ids]
    cf_arr     = np.array([p[1] for p in cf_raw])
    cf_norm    = (cf_arr - cf_arr.min()) / (cf_arr.max() - cf_arr.min() + 1e-9)
    cf_df      = pd.DataFrame({"movieId": [p[0] for p in cf_raw], "cf_score": cf_norm})

    # Combine
    merged = cf_df.merge(cb_df, on="movieId", how="left")
    merged["cb_score"]     = merged["cb_score"].fillna(0)
    merged["hybrid_score"] = cf_weight * merged["cf_score"] + cb_weight * merged["cb_score"]
    merged = merged.sort_values("hybrid_score", ascending=False).head(n)
    result = merged.merge(movies_df[["movieId", "title", "genres"]], on="movieId")
    return result[["title", "genres", "hybrid_score"]].reset_index(drop=True)

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🎬 Movie Recommendation System")
st.caption("Content-Based · Collaborative Filtering (SVD) · Hybrid")

if not models_ready:
    st.warning("⚠️ Model files not found in `./models/`. Run the notebook first (Section 8) to train and save the models.")
    st.code("# In the notebook, run Section 8: Save Models for Streamlit App")
    st.stop()

svd, cosine_sim, cb_data, movies_df, ratings_df = load_models()
all_titles   = sorted(cb_data["title"].unique().tolist())
all_user_ids = sorted(ratings_df["userId"].unique().tolist())

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    mode   = st.radio("Mode", ["🎭 Content-Based", "👥 Collaborative Filtering", "🔀 Hybrid"], index=2)
    n_recs = st.slider("Number of Recommendations", 5, 20, 10)
    if "Hybrid" in mode:
        cb_weight = st.slider("Content-Based Weight", 0.0, 1.0, 0.4, 0.05)
        cf_weight = round(1.0 - cb_weight, 2)
        st.caption(f"CF Weight automatically: {cf_weight}")
    else:
        cb_weight, cf_weight = 0.4, 0.6
    st.markdown("---")
    st.caption("Built with scipy SVD · sklearn TF-IDF · Streamlit")

col1, col2 = st.columns(2)
with col1:
    default_idx = all_titles.index("Toy Story (1995)") if "Toy Story (1995)" in all_titles else 0
    selected_movie = st.selectbox("🎥 Seed Movie (Content-Based / Hybrid)", all_titles, index=default_idx)
with col2:
    selected_user = st.selectbox("👤 User ID (CF / Hybrid)", all_user_ids, index=0)

run_btn = st.button("🚀 Get Recommendations", use_container_width=True, type="primary")

def render_cards(recs, score_col, label, max_score=1.0):
    for _, row in recs.iterrows():
        bar_w = int((row[score_col] / max_score) * 100)
        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-title">{row['title']}</div>
            <div class="rec-genre">{row['genres']}</div>
            <div class="score-bar" style="width:{bar_w}%;"></div>
            <div style="font-size:0.75rem;color:#888;margin-top:4px;">{label}: {row[score_col]:.3f}</div>
        </div>""", unsafe_allow_html=True)

if run_btn:
    with st.spinner("Computing recommendations..."):
        if "Content" in mode:
            recs = cb_recommend(selected_movie, cb_data, cosine_sim, n=n_recs)
            if recs.empty:
                st.error("Movie not found.")
            else:
                st.subheader(f"🎭 Similar to *{selected_movie}*")
                render_cards(recs, "score", "Similarity", max_score=1.0)

        elif "Collaborative" in mode:
            recs = cf_recommend(selected_user, svd, movies_df, ratings_df, n=n_recs)
            st.subheader(f"👥 CF Recommendations for User {selected_user}")
            render_cards(recs, "score", "Predicted Rating",
                         max_score=svd["rating_max"])

        else:
            recs = hybrid_recommend(
                selected_user, selected_movie,
                svd, movies_df, ratings_df, cb_data, cosine_sim,
                n=n_recs, cb_weight=cb_weight, cf_weight=cf_weight
            )
            st.subheader(f"🔀 Hybrid  |  User {selected_user}  ×  *{selected_movie}*")
            render_cards(recs, "hybrid_score", "Hybrid Score", max_score=1.0)

    # Quick stats
    st.markdown("---")
    st.subheader("📊 Dataset Stats")
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, f"{len(movies_df):,}",                  "Movies"),
        (c2, f"{ratings_df['userId'].nunique():,}",   "Users"),
        (c3, f"{len(ratings_df):,}",                  "Ratings"),
        (c4, f"{ratings_df['rating'].mean():.2f}",    "Avg Rating"),
    ]:
        col.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{val}</div>
            <div class="metric-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)
