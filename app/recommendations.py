import os
from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required
from app.robust_movie_recommender import MovieRecommender  # Updated import

BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "combined_movies.2.csv")
OVERVIEW_EMBEDDINGS_PATH = os.path.join(BASE_DIR, "overview_embeddings.npy")

# Instantiate the recommender only one time at startup
recommender = MovieRecommender(CSV_PATH, OVERVIEW_EMBEDDINGS_PATH, device="cpu", use_faiss=True)

recommendations_blueprint = Blueprint("recommendations", __name__)

@recommendations_blueprint.route("/recommendations", methods=["GET"])
@jwt_required()
def get_recommendations():
    title = request.args.get("title")
    if not title:
        return jsonify({"error": "Movie title is required"}), 400

    top_n = request.args.get("top_n", default=5, type=int)
    min_vote = request.args.get("min_vote", default=0.0, type=float)
    plot_weight = request.args.get("plot_weight", default=0.6, type=float)
    genre_weight = request.args.get("genre_weight", default=0.3, type=float)
    sentiment_weight = request.args.get("sentiment_weight", default=0.1, type=float)

    try:
        recs = recommender.recommend(
            movie_title=title,
            top_n=top_n,
            min_vote_average=min_vote,
            plot_weight=plot_weight,
            genre_weight=genre_weight,
            sentiment_weight=sentiment_weight
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not recs:
        return jsonify({"error": f"No recommendations found for '{title}'"}), 404

    recommended_titles = [r[0] for r in recs]
    return jsonify(recommended_titles), 200
