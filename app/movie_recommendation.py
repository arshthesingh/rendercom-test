import os
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity  

# -------------------------------------------------------------------------
# CSV + .npy paths
# -------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "combined_movies.2.csv")

# You can still save/load overview embeddings from disk:
OVERVIEW_EMBEDDINGS_PATH = os.path.join(BASE_DIR, "overview_embeddings.npy")

# -------------------------------------------------------------------------
# Load CSV
# -------------------------------------------------------------------------
movies_data = pd.read_csv(CSV_PATH)
# Fill NaNs
movies_data["overview"] = movies_data["overview"].fillna("")
movies_data["genres"] = movies_data["genres"].fillna("")

# Optionally ensure columns exist
if "title" not in movies_data.columns:
    raise ValueError("CSV must have a 'title' column.")

# -------------------------------------------------------------------------
# Sentence Transformer
# -------------------------------------------------------------------------
sbert_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

# -------------------------------------------------------------------------
# 1) Overview Embeddings
# -------------------------------------------------------------------------
if os.path.exists(OVERVIEW_EMBEDDINGS_PATH):
    print("Loading precomputed overview embeddings from disk...")
    overview_embeddings = np.load(OVERVIEW_EMBEDDINGS_PATH)
else:
    print("Computing overview embeddings (this may take a while)...")
    overview_embeddings = sbert_model.encode(
        movies_data["overview"].tolist(),
        show_progress_bar=True
    )
    np.save(OVERVIEW_EMBEDDINGS_PATH, overview_embeddings)
    print(f"Saved overview embeddings to {OVERVIEW_EMBEDDINGS_PATH}")

# Shape: (N, D)
num_movies, embed_dim = overview_embeddings.shape
print(f"Loaded {num_movies} movie embeddings of dimension {embed_dim}")

# Precompute norms for each movie's embedding
overview_norms = np.linalg.norm(overview_embeddings, axis=1)  # shape (N,)

# -------------------------------------------------------------------------
# 2) Genre One-Hot Encoding
# -------------------------------------------------------------------------
movies_data["genres"] = movies_data["genres"].apply(
    lambda g: g.lower().split(",") if g else []
)
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(movies_data["genres"])  # shape (N, #unique_genres)
genre_norms = np.linalg.norm(genre_encoded, axis=1)  # shape (N,)

# -------------------------------------------------------------------------
# 3) Sentiment Scores
# -------------------------------------------------------------------------
analyzer = SentimentIntensityAnalyzer()
movies_data["sentiment"] = movies_data["overview"].apply(
    lambda text: analyzer.polarity_scores(text)["compound"]
)
# Store as a NumPy array
sentiment = movies_data["sentiment"].values  # shape (N,)
# Precompute norms (for "cosine" in 1D, basically the absolute value)
sentiment_norm = np.abs(sentiment) + 1e-8  # shape (N,)

# -------------------------------------------------------------------------
# Single-pass Cosine Calculation
# -------------------------------------------------------------------------
def single_cosine_sim(
    query_vec: np.ndarray,
    query_norm: float,
    all_vecs: np.ndarray,
    all_norms: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity of `query_vec` vs. each row in `all_vecs`.
    
    :param query_vec: shape (D,)
    :param query_norm: precomputed norm for query_vec
    :param all_vecs: shape (N, D)
    :param all_norms: shape (N,)
    :return: shape (N,) array of similarities
    """
    # Dot products: (N, D) · (D,) -> (N,)
    dot_products = all_vecs @ query_vec  # broadcast
    sims = dot_products / (all_norms * query_norm + 1e-8)
    return sims

def single_cosine_1d(
    query_value: float,
    query_value_abs: float,
    all_values: np.ndarray,
    all_values_abs: np.ndarray
) -> np.ndarray:
    """
    'Cosine similarity' in 1D. For sentiment x and y:
    cos_sim = (x*y)/(|x|*|y| + 1e-8).
    """
    # x*y for each
    dot_products = all_values * query_value
    sims = dot_products / (all_values_abs * query_value_abs + 1e-8)
    return sims

# -------------------------------------------------------------------------
# Main recommendation function (Single-Pass)
# -------------------------------------------------------------------------
def recommend_hybrid(
    movie_title: str,
    top_n: int = 5,
    min_vote_average: float = 0.0,
    plot_weight: float = 0.6,
    genre_weight: float = 0.3,
    sentiment_weight: float = 0.1
):
    """
    Return a list of (title, score) recommended for `movie_title`.
    If `vote_average` is in the CSV, filter out movies below `min_vote_average`.
    We do a single-pass O(N) approach for each request.
    """
    # 1) Find index of the movie
    movie_index = movies_data[
        movies_data["title"].str.lower() == movie_title.lower()
    ].index

    if len(movie_index) == 0:
        return []
    movie_index = movie_index[0]

    # 2) Grab query vectors + norms
    query_plot_vec = overview_embeddings[movie_index]  # shape (D,)
    query_plot_norm = overview_norms[movie_index]

    query_genre_vec = genre_encoded[movie_index]  # shape (#genres,)
    query_genre_norm = genre_norms[movie_index]

    query_sent_value = sentiment[movie_index]
    query_sent_abs = sentiment_norm[movie_index]

    # 3) Compute similarity vs. all movies
    plot_sims = single_cosine_sim(
        query_plot_vec, query_plot_norm, overview_embeddings, overview_norms
    )
    genre_sims = single_cosine_sim(
        query_genre_vec, query_genre_norm, genre_encoded, genre_norms
    )
    sentiment_sims = single_cosine_1d(
        query_sent_value, query_sent_abs, sentiment, sentiment_norm
    )

    # 4) Combine with weights
    combined_sims = (
        plot_weight * plot_sims
        + genre_weight * genre_sims
        + sentiment_weight * sentiment_sims
    )

    # 5) Build a list of (movie_idx, combined_score), skipping the query movie
    #    plus filter by min_vote_average if present
    results = []
    for i in range(num_movies):
        if i == movie_index:
            continue
        if "vote_average" in movies_data.columns:
            if movies_data.iloc[i]["vote_average"] < min_vote_average:
                continue

        score = combined_sims[i]
        results.append((i, score))

    # 6) Sort by combined score descending
    results.sort(key=lambda x: x[1], reverse=True)

    # 7) Take top_n
    top_results = results[:top_n]

    # 8) Convert indices to (title, score)
    recommendations = []
    for idx, score in top_results:
        recommendations.append(
            (movies_data.iloc[idx]["title"], float(score))
        )

    return recommendations



