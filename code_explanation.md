# Code Explanation: `build_model.py`

This document explains the code line-by-line so you understand exactly what is happening under the hood.

## 1. Importing Libraries
```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.stem.porter import PorterStemmer
```
*   **`pandas`**: Like Excel for Python. It handles tables of data (rows and columns).
*   **`numpy`**: Handles math and big arrays of numbers efficiently.
*   **`TfidfVectorizer`**: The tool that converts text into numbers (vectors).
*   **`re`**: "Regular Expressions". Used for advanced find-and-replace in text.
*   **`PorterStemmer`**: A tool that chops words to their root (e.g., "running" -> "run").

## 2. Helper Functions

### `process_text(text)`
```python
def process_text(text):
    if not isinstance(text, str):
        return []
    tokens = re.split(r'[\s,]+', text.lower())
    return [token for token in tokens if token]
```
*   **Goal**: Clean up messy text.
*   **How**:
    1.  Checks if the input is actually text.
    2.  `text.lower()`: Converts "Iron Man" to "iron man" (so computer knows they are the same).
    3.  `re.split`: Splits the sentence into a list of words, removing spaces and commas.
    4.  Returns a clean list of words.

### `stem(text)`
```python
def stem(text):
    ps = PorterStemmer()
    return " ".join(ps.stem(word) for word in text.split())
```
*   **Goal**: Simplify words so "act", "acting", and "actor" are treated as the same concept.
*   **How**: It loops through every word and cuts off the ending.
    *   *Input*: "loving the movies"
    *   *Output*: "love the movi"

## 3. The Main Function: `build_model()`

### Loading and Filtering
```python
print("Loading data...")
movies = pd.read_csv("TMDB.csv")

# Filter by vote count
movies = movies[movies['vote_count'] > 500]

# Exclude documentaries
movies = movies[~movies['genres'].str.contains('Documentary', na=False)]
```
*   **`read_csv`**: Opens your big data file.
*   **`movies['vote_count'] > 500`**: Keeps only rows where the 'vote_count' column is bigger than 500.
*   **`~movies['genres'].str.contains('Documentary')`**: The `~` symbol means "NOT". So this keeps rows that do *not* contain the word "Documentary".

### Cleaning Columns
```python
movies = movies[["id","title","overview", ... "keywords"]]
movies = movies[movies['overview'].notna()]
movies.loc[:, 'tagline'] = movies['tagline'].fillna("")
```
*   We select only the columns we need.
*   **`notna()`**: Removes movies that have no description (overview).
*   **`fillna("")`**: If a movie has no tagline, we replace the empty space (NaN) with an empty string `""` so the code doesn't crash later.

### Creating the "Soup" (Feature Engineering)
```python
movies.loc[:, 'overview_tokens'] = movies['overview'].apply(process_text)
# ... (repeats for genres, keywords, companies)

movies.loc[:, 'encode_list'] = (
    movies['overview_tokens'] + 
    movies['genres_tokens'] + 
    movies['companies_tokens'] + 
    movies['keywords_tokens']
).apply(lambda tokens: ' '.join(tokens))
```
*   **`apply(process_text)`**: Runs our helper function on every single row.
*   **`+`**: Joins the lists together. We are mashing up the description, genre, and keywords into one giant list of words for each movie.
*   **`' '.join(tokens)`**: Stitches the list of words back into a single string of text.

### Vectorization (The Math Part)
```python
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
features = vectorizer.fit_transform(new_movies['encode_list']).toarray()
```
*   **`max_features=5000`**: We tell it to only learn the top 5,000 most important words. If we used every unique word, the model would be too huge and slow.
*   **`stop_words='english'`**: Tells it to ignore useless words like "the", "a", "in".
*   **`fit_transform`**: This does two things:
    1.  **Fit**: Learns the vocabulary (finds the top 5,000 words).
    2.  **Transform**: Converts every movie's text into a list of numbers based on that vocabulary.

### Saving the Results
```python
np.savez_compressed('feature_array.npz', feature_array)
new_movies.to_pickle('movies_df.pkl')
```
*   We save the data so we don't have to re-calculate it every time we want a recommendation.
*   **`.npz`**: A compressed file for the number arrays (efficient).
*   **`.pkl`**: A "pickle" file. It saves the Python dataframe exactly as is, so we can load it back easily.

---
## Summary
1.  **Load** raw data.
2.  **Filter** out bad data.
3.  **Clean** the text (lowercase, stemming).
4.  **Combine** text into one "soup".
5.  **Vectorize** (Text -> Numbers).
6.  **Save** for later use.

## 4. The Recommendation Script: `recommend.py`

This script is what you run to actually *get* recommendations. It uses the files created by `build_model.py`.

### Loading the Saved Data
```python
movies = pd.read_pickle('movies_df.pkl')
feature_array = np.load('feature_array.npz')['arr_0']
```
*   Instead of processing the CSV again (which takes time), we load the ready-made `.pkl` (movie list) and `.npz` (number vectors) files.
*   This is very fast!

### Finding the Movie
```python
matches = movies[movies['title'].str.contains(title_query, case=False, na=False)]
movie_idx = matches.index[0]
```
*   **`str.contains`**: Searches for the movie title you typed.
*   **`case=False`**: Makes it case-insensitive (so "avengers" finds "The Avengers").
*   **`matches.index[0]`**: We take the *index* (ID) of the first matching movie found. We need this ID to find its corresponding vector.

### Calculating Similarity
```python
movie_vector = feature_array[movie_idx].reshape(1, -1)
similarity_scores = cosine_similarity(movie_vector, feature_array).flatten()
```
*   **`movie_vector`**: We grab the row of numbers for *our* movie (e.g., The Avengers).
*   **`cosine_similarity`**: This is a math function from `sklearn`. It compares our `movie_vector` against `feature_array` (ALL movies).
*   **Result**: It returns a list of scores between 0 and 1 for every movie.
    *   `1.0` = Exact match (The movie itself).
    *   `0.0` = Completely different.

### Sorting and Printing
```python
similar_indices = similarity_scores.argsort()[-11:-1][::-1]
```
*   **`argsort()`**: Sorts the scores from lowest to highest and returns their *indices* (IDs).
*   **`[-11:-1]`**: We take the last 11 items (which are the highest scores). We stop at `-1` to exclude the movie itself (which is always the #1 match).
*   **`[::-1]`**: Reverses the list so the *highest* score comes first.

Finally, the loop prints out the titles corresponding to these top indices.
