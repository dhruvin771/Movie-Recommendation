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
    
    # Return top 10 matches
    suggestions = matches.head(10)['title'].tolist()
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
