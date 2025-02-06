import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MultiLabelBinarizer

class MovieRecommender:
    def __init__(self, csv_path: str, embeddings_path: str, device: str = "cpu", use_faiss: bool = True):
        """
        Initialize with paths and settings. Note that we defer loading heavy resources (the model and embeddings)
        until they are needed.
        """
        self.csv_path = csv_path
        self.embeddings_path = embeddings_path
        self.device = device
        self.use_faiss = use_faiss

        # Load basic movie data and process genres
        self.movies_data = pd.read_csv(csv_path)
        self.movies_data["overview"] = self.movies_data["overview"].fillna("")
        self.movies_data["genres"] = self.movies_data["genres"].fillna("")
        if "title" not in self.movies_data.columns:
            raise ValueError("CSV must have a 'title' column.")

        # Process genres
        self.movies_data["genres_list"] = self.movies_data["genres"].apply(
            lambda g: [genre.strip().lower() for genre in g.split(",") if genre.strip()]
        )
        self.mlb = MultiLabelBinarizer()
        self.genre_encoded = self.mlb.fit_transform(self.movies_data["genres_list"])
        self.genre_norms = np.linalg.norm(self.genre_encoded, axis=1, keepdims=True) + 1e-8
        self.genre_normalized = self.genre_encoded / self.genre_norms

        # Initialize heavy resources to None for lazy loading
        self.sbert_model = None
        self.sentiment_analyzer = None
        self.overview_embeddings = None
        self.overview_normalized = None
        self.faiss_index = None

    def _lazy_load_resources(self):
        """
        Lazy-load heavy resources (the transformer model, sentiment analyzer, embeddings, and FAISS index)
        only when they are needed.
        """
        if self.sbert_model is None:
            print("Loading transformer model...")
            self.sbert_model = SentenceTransformer("all-mpnet-base-v2", device=self.device)
            print("Loading sentiment analyzer...")
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        if self.overview_embeddings is None or self.overview_normalized is None:
            print("Loading and processing embeddings lazily...")
            self._compute_overview_embeddings()
            if self.use_faiss:
                dim = self.overview_normalized.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dim)
                self.faiss_index.add(self.overview_normalized.astype('float32'))
    
    def _compute_overview_embeddings(self):
        """
        Compute or load precomputed embeddings and normalize them using memory mapping and chunked processing.
        """
        if os.path.exists(self.embeddings_path):
            print("Loading precomputed overview embeddings with memory mapping...")
            overview_embeddings = np.load(self.embeddings_path, mmap_mode='r')
        else:
            print("Computing overview embeddings (this may take a while)...")
            overview_embeddings = self.sbert_model.encode(
                self.movies_data["overview"].tolist(),
                show_progress_bar=True
            )
            np.save(self.embeddings_path, overview_embeddings)
            overview_embeddings = np.load(self.embeddings_path, mmap_mode='r')
        
        self.overview_embeddings = overview_embeddings
        self.overview_normalized = self._normalize_embeddings_in_chunks(overview_embeddings)

    def _normalize_embeddings_in_chunks(self, embeddings, chunk_size=1000):
        """
        Normalize embeddings in chunks to reduce peak memory usage.
        """
        num_rows = embeddings.shape[0]
        normalized = np.empty(embeddings.shape, dtype=np.float32)
        for start in range(0, num_rows, chunk_size):
            end = min(start + chunk_size, num_rows)
            # Convert the current chunk into a regular NumPy array if it's a memmap slice
            chunk = np.array(embeddings[start:end])
            norms = np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-8
            normalized[start:end] = chunk / norms
        return normalized

    def recommend(self,
                  movie_title: str,
                  top_n: int = 5,
                  min_vote: float = 0.0,
                  plot_weight: float = 0.6,
                  genre_weight: float = 0.3,
                  sentiment_weight: float = 0.1,
                  faiss_candidate_pool: int = 50):
        """
        Generate recommendations for a given movie title. This method first ensures that all heavy resources are loaded.
        """
        # Lazy-load heavy resources on first use.
        self._lazy_load_resources()

        # Retrieve query index (raise an error if not found)
        query_idx = self._get_movie_index(movie_title)

        # Retrieve query vectors for overview, genre, and sentiment
        query_overview = self.overview_normalized[query_idx]
        query_genre = self.genre_normalized[query_idx]
        query_sentiment = self.movies_data.iloc[query_idx].get("sentiment", 0.0)
        query_sent_norm = np.abs(query_sentiment) + 1e-8

        # Use FAISS for candidate selection if enabled; otherwise, consider all movies.
        if self.use_faiss:
            query_vector = np.array([query_overview], dtype='float32')
            _, candidate_indices = self.faiss_index.search(query_vector, faiss_candidate_pool)
            candidate_indices = candidate_indices[0].tolist()
        else:
            candidate_indices = list(range(len(self.movies_data)))

        # Exclude the query movie itself.
        if query_idx in candidate_indices:
            candidate_indices.remove(query_idx)

        recommendations = []
        for idx in candidate_indices:
            sim_overview = np.dot(query_overview, self.overview_normalized[idx])
            sim_genre = np.dot(query_genre, self.genre_normalized[idx])
            candidate_sent = self.movies_data.iloc[idx].get("sentiment", 0.0)
            candidate_sent_norm = np.abs(candidate_sent) + 1e-8
            sim_sentiment = (query_sentiment * candidate_sent) / (query_sent_norm * candidate_sent_norm)
            combined_sim = (plot_weight * sim_overview +
                            genre_weight * sim_genre +
                            sentiment_weight * sim_sentiment)
            # (Optionally, apply additional filtering based on other movie attributes)
            recommendations.append((idx, combined_sim))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = recommendations[:top_n]
        result = [(self.movies_data.iloc[idx]["title"], score) for idx, score in top_recommendations]
        return result

    def _get_movie_index(self, title: str) -> int:
        """
        Retrieve the index of a movie by title.
        """
        idx = self.movies_data[self.movies_data["title"].str.lower() == title.lower()].index
        if len(idx) == 0:
            raise ValueError(f"Movie title '{title}' not found.")
        return idx[0]
