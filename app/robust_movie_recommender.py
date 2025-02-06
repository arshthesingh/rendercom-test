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
        Initialize the recommender by loading data, computing embeddings, and setting up FAISS.
        """
        self.csv_path = csv_path
        self.embeddings_path = embeddings_path
        self.device = device
        self.use_faiss = use_faiss

        # Preprocess movie data
        self.movies_data = pd.read_csv(csv_path)
        self.movies_data["overview"] = self.movies_data["overview"].fillna("")
        self.movies_data["genres"] = self.movies_data["genres"].fillna("")
        if "title" not in self.movies_data.columns:
            raise ValueError("CSV must have 'title' column.")

        # Sentence Transformer and Sentiment Analyzer
        self.sbert_model = SentenceTransformer("all-mpnet-base-v2", device=device)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Compute or load overview embeddings
        self._compute_overview_embeddings()

        # Prepare FAISS index (using cosine similarity via inner product on normalized vectors)
        self._prepare_faiss_index()

        # Process genres (convert to lower-case and split)
        self.movies_data["genres_list"] = self.movies_data["genres"].apply(
            lambda g: [genre.strip().lower() for genre in g.split(",") if genre.strip()]
        )
        self.mlb = MultiLabelBinarizer()
        self.genre_encoded = self.mlb.fit_transform(self.movies_data["genres_list"])
        # Normalize genre vectors
        self.genre_norms = np.linalg.norm(self.genre_encoded, axis=1, keepdims=True) + 1e-8
        self.genre_normalized = self.genre_encoded / self.genre_norms

        # Compute sentiment scores from the movie overview text
        self.movies_data["sentiment"] = self.movies_data["overview"].apply(
            lambda text: self.sentiment_analyzer.polarity_scores(text)["compound"]
        )
        self.sentiment = self.movies_data["sentiment"].values
        self.sentiment_norm = (np.abs(self.sentiment) + 1e-8)

    def _compute_overview_embeddings(self):
        """
        Compute or load the precomputed overview embeddings and normalize them
        using memory mapping and chunked processing.
        """
        if os.path.exists(self.embeddings_path):
            print("Loading precomputed overview embeddings with memory mapping...")
            # Load the embeddings using memory mapping (read-only)
            overview_embeddings = np.load(self.embeddings_path, mmap_mode='r')
        else:
            print("Computing overview embeddings (this may take a while)...")
            overview_embeddings = self.sbert_model.encode(
                self.movies_data["overview"].tolist(),
                show_progress_bar=True
            )
            # Save the computed embeddings for future runs
            np.save(self.embeddings_path, overview_embeddings)
            # Re-load the saved embeddings using memory mapping
            overview_embeddings = np.load(self.embeddings_path, mmap_mode='r')
        
        # Store the loaded embeddings if you need them later
        self.overview_embeddings = overview_embeddings

        # Normalize the embeddings in manageable chunks to reduce memory usage
        self.overview_normalized = self._normalize_embeddings_in_chunks(overview_embeddings)


    def _normalize_embeddings_in_chunks(self, embeddings, chunk_size=1000):
        """
        Normalize embeddings in chunks to avoid allocating a huge array all at once.

        Parameters:
            embeddings (np.array or memmap): The embeddings to normalize.
            chunk_size (int): Number of rows to process at a time.

        Returns:
            np.array: A fully normalized array of embeddings.
        """
        num_rows = embeddings.shape[0]
        # Preallocate an array for normalized embeddings (using float32 for lower memory usage)
        normalized = np.empty(embeddings.shape, dtype=np.float32)
        
        for start in range(0, num_rows, chunk_size):
            end = min(start + chunk_size, num_rows)
            # Convert the chunk to a regular NumPy array if it's still memory mapped
            chunk = np.array(embeddings[start:end])
            # Compute the L2 norms and add a small epsilon to avoid division by zero
            norms = np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-8
            normalized[start:end] = chunk / norms
        return normalized


    def _prepare_faiss_index(self):
        """Build a FAISS index on the normalized overview embeddings."""
        if self.use_faiss:
            dim = self.overview_normalized.shape[1]
            # For normalized vectors, inner product is equivalent to cosine similarity.
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.overview_normalized.astype('float32'))

    def _get_movie_index(self, title: str) -> int:
        """Retrieve the index of the movie by title."""
        idx = self.movies_data[self.movies_data["title"].str.lower() == title.lower()].index
        if len(idx) == 0:
            raise ValueError(f"Movie title '{title}' not found.")
        return idx[0]

    def recommend(self,
                  movie_title: str,
                  top_n: int = 5,
                  min_vote_average: float = 0.0,
                  plot_weight: float = 0.6,
                  genre_weight: float = 0.3,
                  sentiment_weight: float = 0.1,
                  faiss_candidate_pool: int = 50):
        """
        Return a list of (title, score) recommendations for the given movie title.
        Uses FAISS for an initial candidate search on overview embeddings, then re-ranks using
        a weighted combination of overview, genre, and sentiment similarities.
        """
        try:
            query_idx = self._get_movie_index(movie_title)
        except ValueError as e:
            print(e)
            return []

        # Retrieve query vectors
        query_overview = self.overview_normalized[query_idx]
        query_genre = self.genre_normalized[query_idx]
        query_sentiment = self.sentiment[query_idx]
        query_sent_norm = np.abs(query_sentiment) + 1e-8

        # Use FAISS to get candidate movies based on overview similarity
        if self.use_faiss:
            query_vector = np.array([query_overview], dtype='float32')
            _, candidate_indices = self.faiss_index.search(query_vector, faiss_candidate_pool)
            candidate_indices = candidate_indices[0].tolist()
        else:
            candidate_indices = list(range(len(self.movies_data)))

        # Exclude the query movie itself from candidates
        if query_idx in candidate_indices:
            candidate_indices.remove(query_idx)

        recommendations = []
        for idx in candidate_indices:
            # Overview similarity (cosine similarity via inner product)
            sim_overview = np.dot(query_overview, self.overview_normalized[idx])
            # Genre similarity
            sim_genre = np.dot(query_genre, self.genre_normalized[idx])
            # Sentiment similarity (using 1D cosine similarity)
            candidate_sent = self.sentiment[idx]
            candidate_sent_norm = np.abs(candidate_sent) + 1e-8
            sim_sentiment = (query_sentiment * candidate_sent) / (query_sent_norm * candidate_sent_norm)

            # Combine similarities using the provided weights
            combined_sim = (plot_weight * sim_overview +
                            genre_weight * sim_genre +
                            sentiment_weight * sim_sentiment)

            # Optionally filter by vote_average if that column exists
            if "vote_average" in self.movies_data.columns:
                if self.movies_data.iloc[idx]["vote_average"] < min_vote_average:
                    continue

            recommendations.append((idx, combined_sim))

        # Sort the recommendations by the combined similarity score (highest first)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = recommendations[:top_n]
        result = [(self.movies_data.iloc[idx]["title"], score) for idx, score in top_recommendations]
        return result

# Example usage:
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__)
    CSV_PATH = os.path.join(BASE_DIR, "combined_movies.2.csv")
    OVERVIEW_EMBEDDINGS_PATH = os.path.join(BASE_DIR, "overview_embeddings.npy")

    recommender = MovieRecommender(CSV_PATH, OVERVIEW_EMBEDDINGS_PATH, device="cpu", use_faiss=True)
    movie_title = "The Matrix"  # Replace with any movie title present in your CSV
    recommendations = recommender.recommend(movie_title)
    
    print(f"Recommendations for '{movie_title}':")
    for title, score in recommendations:
        print(f" - {title}: {score:.4f}")
