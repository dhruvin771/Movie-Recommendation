import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.stem.porter import PorterStemmer
import nltk
import os
from huggingface_hub import hf_hub_download

# Download nltk resources if not present (though likely are)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def process_text(text):
    if not isinstance(text, str):
        return []
    # Remove punctuation and split by spaces or commas
    tokens = re.split(r'[\s,]+', text.lower())
    # Remove empty strings
    return [token for token in tokens if token]

def stem(text):
    ps = PorterStemmer()
    return " ".join(ps.stem(word) for word in text.split())

def get_dataset():
    filename = "TMDB_movie_dataset_v11.csv"
    if os.path.exists(filename):
        print(f"Found {filename} locally.")
        return filename
    
    print(f"Downloading {filename} from Hugging Face...")
    try:
        downloaded_path = hf_hub_download(
            repo_id="eka416/movies",
            filename="TMDB_movie_dataset_v11.csv",
            repo_type="dataset",
            local_dir=".",
            local_dir_use_symlinks=False
        )
        print(f"Downloaded to {downloaded_path}")
        return filename
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        # Fallback to original if download fails, or raise error
        if os.path.exists("TMDB.csv"):
            print("Falling back to TMDB.csv")
            return "TMDB.csv"
        raise

def build_model():
    print("Loading data...")
    csv_file = get_dataset()
    movies = pd.read_csv(csv_file)
    print(f"Original shape: {movies.shape}")
    
    # --- IMPROVEMENT 1: Filter by vote count ---
    print("Filtering movies with vote_count > 500...")
    movies = movies[movies['vote_count'] > 500]
    print(f"After vote filter: {movies.shape}")
    
    # --- IMPROVEMENT 2: Exclude documentaries ---
    print("Excluding documentaries...")
    movies = movies[~movies['genres'].str.contains('Documentary', na=False)]
    print(f"After documentary filter: {movies.shape}")
    
    # Keep necessary columns
    movies = movies[["id","title","overview","poster_path","backdrop_path","release_date","tagline","production_companies","genres","keywords"]]
    
    # Drop rows with missing overview
    movies = movies[movies['overview'].notna()]
    
    # Drop backdrop_path
    movies.drop(columns=["backdrop_path"], inplace=True, errors='ignore')
    
    # Fill NA
    movies.loc[:, 'tagline'] = movies['tagline'].fillna("")
    movies.loc[:, 'keywords'] = movies['keywords'].fillna("")
    movies.loc[:, 'production_companies'] = movies['production_companies'].fillna("")
    movies.loc[:, 'genres'] = movies['genres'].fillna("")
    
    print("Processing text columns...")
    # Process columns
    movies.loc[:, 'overview_tokens'] = movies['overview'].apply(process_text)
    movies.loc[:, 'genres_tokens'] = movies['genres'].apply(process_text)
    movies.loc[:, 'keywords_tokens'] = movies['keywords'].apply(process_text)
    movies.loc[:, 'companies_tokens'] = movies['production_companies'].apply(process_text)
    
    # --- IMPROVEMENT 3: Add franchise/series detection ---
    # Extract base title (before colon) to help sequels be recommended together
    def extract_franchise(title):
        if not isinstance(title, str):
            return []
        # Get the part before the colon (e.g., "Avatar" from "Avatar: The Way of Water")
        base_title = title.split(':')[0].strip()
        # Also check for common sequel patterns
        base_title = re.sub(r'\s+(Part|Chapter|Episode|Volume|Book)\s+\d+', '', base_title, flags=re.IGNORECASE)
        base_title = re.sub(r'\s+\d+$', '', base_title)  # Remove trailing numbers like "Avatar 2"
        
        # CRITICAL FIX: Join with underscores to create a UNIQUE token
        # This prevents "Iron Man" (Iron, Man) from matching "Iron Lady" (Iron, Lady)
        # It will now be "franchise_Iron_Man" which only matches other "franchise_Iron_Man"
        slug = "franchise_" + base_title.lower().replace(' ', '_')
        return [slug]
    
    movies.loc[:, 'franchise_tokens'] = movies['title'].apply(extract_franchise)
    
    # Combine tokens - give extra weight to franchise and companies
    movies.loc[:, 'encode_list'] = (
        movies['overview_tokens'] + 
        movies['genres_tokens'] + 
        movies['companies_tokens'] * 2 +  # Double weight for Studios (Marvel, etc.)
        movies['keywords_tokens'] +
        movies['franchise_tokens'] * 5  # High weight for exact franchise match
    ).apply(lambda tokens: ' '.join(tokens))
    
    print("Stemming...")
    # Stemming
    movies.loc[:, 'encode_list'] = movies['encode_list'].map(stem)
    
    # Create final dataframe - also save release_date for the frontend
    new_movies = movies[['id', 'title', 'poster_path', 'release_date', 'encode_list']].copy()
    
    # --- IMPROVEMENT 2: Increase max_features ---
    print("Vectorizing with max_features=5000...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    features = vectorizer.fit_transform(new_movies['encode_list']).toarray()
    
    print("Saving artifacts...")
    # Save features
    feature_array = features.astype('float32')
    np.savez_compressed('feature_array.npz', feature_array)
    
    # Save dataframe (pickle for preservation of types/index, csv for compatibility)
    new_movies.to_pickle('movies_df.pkl')
    new_movies.to_csv('movies_df.csv', index=False)
    
    print("Build complete.")

if __name__ == "__main__":
    build_model()
