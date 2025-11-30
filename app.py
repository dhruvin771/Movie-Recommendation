from flask import Flask, request, jsonify
from flask_cors import CORS
from recommend import get_recommendations
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load movies data once at startup for search
movies_df = None

def load_movies():
    global movies_df
    if movies_df is None:
        try:
            movies_df = pd.read_pickle('movies_df.pkl')
        except FileNotFoundError:
            pass
    return movies_df

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '').strip()
    if not query or len(query) < 2:
        return jsonify({"suggestions": []})
    
    df = load_movies()
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500
    
    # Search for matching titles (case-insensitive)
    matches = df[df['title'].str.contains(query, case=False, na=False)]
    
    # Return top 10 matches with title, year, and poster
    suggestions = []
    for _, row in matches.head(10).iterrows():
        title = row['title']
        # Extract year from release_date if available
        year = ''
        if 'release_date' in row and pd.notna(row['release_date']):
            try:
                year = str(row['release_date'])[:4]  # Get first 4 characters (year)
            except:
                pass
        
        # Get poster path
        poster_path = ''
        if 'poster_path' in row and pd.notna(row['poster_path']):
            poster_path = "https://image.tmdb.org/t/p/w92/" + row['poster_path']  # Small size for dropdown
        
        suggestions.append({
            "title": title,
            "year": year,
            "poster_path": poster_path,
            "display": f"{title} ({year})" if year else title
        })
    
    return jsonify({"suggestions": suggestions})

@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    if not title:
        return jsonify({"error": "Please provide a movie title"}), 400
    
    result = get_recommendations(title)
    if "error" in result:
        return jsonify(result), 404
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
