import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(title_query):
    print(f"Loading data...")
    try:
        movies = pd.read_pickle('movies_df.pkl')
        feature_array = np.load('feature_array.npz')['arr_0']
    except FileNotFoundError:
        print("Data files not found. Please run build_model.py first to generate .pkl and .npz files.")
        return

    print(f"Data loaded. Movies shape: {movies.shape}, Features shape: {feature_array.shape}")

    # Find the movie index
    # Case-insensitive search
    matches = movies[movies['title'].str.contains(title_query, case=False, na=False)]
    
    if matches.empty:
        print(f"No movie found matching '{title_query}'")
        return

    # Use the first match
    movie_idx = matches.index[0]
    movie_title = matches.iloc[0]['title']
    print(f"Found movie: '{movie_title}' (Index: {movie_idx})")

    # Calculate similarity
    movie_vector = feature_array[movie_idx].reshape(1, -1)
    similarity_scores = cosine_similarity(movie_vector, feature_array).flatten()
    
    # Get top 10 recommendations
    similar_indices = similarity_scores.argsort()[-11:-1][::-1]
    
    print(f"\nRecommendations for '{movie_title}':")
    for i, idx in enumerate(similar_indices):
        rec_title = movies.iloc[idx]['title']
        rec_score = similarity_scores[idx]
        print(f"{i+1}. {rec_title} (Score: {rec_score:.4f})")

if __name__ == "__main__":
    get_recommendations("Captain America")
