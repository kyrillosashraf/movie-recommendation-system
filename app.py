"""
app.py  –  Movie Recommendation System Streamlit UI
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# ── CSS styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .rec-card {
        background: linear-gradient(135deg, #1e2130, #252a40);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 10px;
        border-left: 4px solid #e50914;
    }
    .rec-title { font-size: 1.05rem; font-weight: 700; color: #f0f0f0; }
    .rec-genre { font-size: 0.8rem;  color: #aaa; margin-top: 4px; }
    .score-bar { height: 6px; border-radius: 3px; background: #e50914; margin-top: 8px; }
    .metric-box {
        background: #1e2130; border-radius: 10px;
        padding: 18px; text-align: center;
    }
    .metric-val { font-size: 2rem; font-weight: 800; color: #e50914; }
    .metric-lbl { font-size: 0.85rem; color: #aaa; }
</style>
""", unsafe_allow_html=True)


# ── Load artefacts ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    base = Path("models")
    with open(base / "svd_model.pkl",  "rb") as f: svd      = pickle.load(f)
    with open(base / "cosine_sim.pkl", "rb") as f: cos_sim  = pickle.load(f)
    cb_data  = pd.read_pickle(base / "cb_data.pkl")
    movies   = pd.read_pickle(base / "movies.pkl")
    ratings  = pd.read_pickle(base / "ratings.pkl")
    return svd, cos_sim, cb_data, movies, ratings


# Check models exist before loading
models_ready = all(
    os.path.exists(f"models/{fn}")
    for fn in ["svd_model.pkl", "cosine_sim.pkl", "cb_data.pkl", "movies.pkl", "ratings.pkl"]
)


# ── Recommendation helpers ─────────────────────────────────────────────────────
def cb_recommend(title, cb_data, cosine_sim, n=10):
    indices = pd.Series(cb_data.index, index=cb_data['title'])
    if title not in indices:
        return pd.DataFrame()
    idx    = indices[title]
    scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:n+1]
    movie_idx  = [i[0] for i in scores]
    sim_values = [i[1] for i in scores]
    result = cb_data[['movieId', 'title', 'genres']].iloc[movie_idx].copy()
    result['score'] = sim_values
    return result.reset_index(drop=True)


def cf_recommend(user_id, svd, movies, ratings, n=10):
    rated_ids  = set(ratings[ratings['userId'] == user_id]['movieId'])
    all_ids    = set(movies['movieId'])
    unseen_ids = list(all_ids - rated_ids)
    preds      = [(mid, svd.predict(user_id, mid).est) for mid in unseen_ids]
    preds.sort(key=lambda x: x[1], reverse=True)
    top_ids    = [p[0] for p in preds[:n]]
    top_scores = [p[1] for p in preds[:n]]
    result     = movies[movies['movieId'].isin(top_ids)][['movieId', 'title', 'genres']].copy()
    score_map  = dict(zip(top_ids, top_scores))
    result['score'] = result['movieId'].map(score_map)
    return result.sort_values('score', ascending=False).reset_index(drop=True)


def hybrid_recommend(user_id, title, svd, movies, ratings, cb_data, cosine_sim,
                     n=10, cb_weight=0.4, cf_weight=0.6):
    # Content-based scores
    cb = cb_recommend(title, cb_data, cosine_sim, n=500)
    if not cb.empty:
        cb_max = cb['score'].max() + 1e-9
        cb['cb_score'] = cb['score'] / cb_max
    else:
        cb = pd.DataFrame(columns=['movieId', 'cb_score'])

    # CF scores
    rated_ids  = set(ratings[ratings['userId'] == user_id]['movieId'])
    all_ids    = set(movies['movieId'])
    unseen_ids = list(all_ids - rated_ids)
    preds      = [(mid, svd.predict(user_id, mid).est) for mid in unseen_ids]
    pred_arr   = np.array([p[1] for p in preds])
    mn, mx     = pred_arr.min(), pred_arr.max()
    cf_norm    = (pred_arr - mn) / (mx - mn + 1e-9)
    cf_df      = pd.DataFrame({'movieId': [p[0] for p in preds], 'cf_score': cf_norm})

    # Merge
    merged = cf_df.merge(cb[['movieId', 'cb_score']], on='movieId', how='left')
    merged['cb_score']     = merged['cb_score'].fillna(0)
    merged['hybrid_score'] = cf_weight * merged['cf_score'] + cb_weight * merged['cb_score']
    merged = merged.sort_values('hybrid_score', ascending=False).head(n)
    result = merged.merge(movies[['movieId', 'title', 'genres']], on='movieId')
    result = result[['title', 'genres', 'hybrid_score']].reset_index(drop=True)
    return result


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🎬 Movie Recommendation System")
st.caption("Content-Based · Collaborative Filtering · Hybrid")

if not models_ready:
    st.warning("⚠️ Model files not found in `./models/`. Please run the notebook first to train and save the models.")
    st.code("# In the notebook, run Section 8: Save Models for Streamlit App", language="python")
    st.stop()

svd, cosine_sim, cb_data, movies_df, ratings_df = load_models()
all_titles  = sorted(cb_data['title'].unique().tolist())
all_user_ids = sorted(ratings_df['userId'].unique().tolist())

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio(
        "Recommendation Mode",
        ["🎭 Content-Based", "👥 Collaborative Filtering", "🔀 Hybrid"],
        index=2,
    )
    n_recs = st.slider("Number of Recommendations", 5, 20, 10)

    if mode in ["🔀 Hybrid"]:
        st.markdown("**Hybrid Weights**")
        cb_weight = st.slider("Content-Based Weight", 0.0, 1.0, 0.4, 0.05)
        cf_weight = round(1.0 - cb_weight, 2)
        st.caption(f"CF Weight automatically: {cf_weight}")
    else:
        cb_weight, cf_weight = 0.4, 0.6

    st.markdown("---")
    st.markdown("**About**")
    st.caption("Built with Surprise (SVD), scikit-learn (TF-IDF), and Streamlit.")

# ── Main input area ────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    selected_movie = st.selectbox(
        "🎥 Select a seed movie (for Content-Based / Hybrid)",
        all_titles,
        index=all_titles.index("Toy Story (1995)") if "Toy Story (1995)" in all_titles else 0,
    )

with col2:
    selected_user = st.selectbox(
        "👤 Select a User ID (for CF / Hybrid)",
        all_user_ids,
        index=0,
    )

run_btn = st.button("🚀 Get Recommendations", use_container_width=True, type="primary")

# ── Results ────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Computing recommendations..."):

        if mode == "🎭 Content-Based":
            recs = cb_recommend(selected_movie, cb_data, cosine_sim, n=n_recs)
            if recs.empty:
                st.error("Movie not found in the database.")
            else:
                st.subheader(f"🎭 Content-Based: Movies similar to *{selected_movie}*")
                for _, row in recs.iterrows():
                    bar_w = int(row['score'] * 100)
                    st.markdown(f"""
                    <div class="rec-card">
                        <div class="rec-title">{row['title']}</div>
                        <div class="rec-genre">{row['genres']}</div>
                        <div class="score-bar" style="width:{bar_w}%;"></div>
                        <div style="font-size:0.75rem;color:#888;margin-top:4px;">Similarity: {row['score']:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)

        elif mode == "👥 Collaborative Filtering":
            recs = cf_recommend(selected_user, svd, movies_df, ratings_df, n=n_recs)
            st.subheader(f"👥 CF Recommendations for User {selected_user}")
            for _, row in recs.iterrows():
                bar_w = int((row['score'] / 5.0) * 100)
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-title">{row['title']}</div>
                    <div class="rec-genre">{row['genres']}</div>
                    <div class="score-bar" style="width:{bar_w}%;"></div>
                    <div style="font-size:0.75rem;color:#888;margin-top:4px;">Predicted Rating: {row['score']:.2f} / 5.0</div>
                </div>
                """, unsafe_allow_html=True)

        else:  # Hybrid
            recs = hybrid_recommend(
                selected_user, selected_movie,
                svd, movies_df, ratings_df, cb_data, cosine_sim,
                n=n_recs, cb_weight=cb_weight, cf_weight=cf_weight
            )
            st.subheader(f"🔀 Hybrid Recommendations  |  User {selected_user}  ×  *{selected_movie}*")
            for _, row in recs.iterrows():
                bar_w = int(row['hybrid_score'] * 100)
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-title">{row['title']}</div>
                    <div class="rec-genre">{row['genres']}</div>
                    <div class="score-bar" style="width:{bar_w}%;"></div>
                    <div style="font-size:0.75rem;color:#888;margin-top:4px;">Hybrid Score: {row['hybrid_score']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Quick stats ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Quick Dataset Stats")
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, f"{len(movies_df):,}", "Movies"),
        (c2, f"{ratings_df['userId'].nunique():,}", "Users"),
        (c3, f"{len(ratings_df):,}", "Ratings"),
        (c4, f"{ratings_df['rating'].mean():.2f}", "Avg Rating"),
    ]:
        col.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{val}</div>
            <div class="metric-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)
