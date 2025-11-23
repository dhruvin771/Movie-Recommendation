# Movie Recommendation System - Beginner's Guide

Welcome to your first Machine Learning project! This guide explains how your movie recommender works in simple English.

## 🎯 The Goal
Imagine you have a friend who knows every movie ever made. You tell them, "I liked *The Avengers*", and they say, "Oh, you should watch *Guardians of the Galaxy*!"

This project builds that "friend" using code. It looks at a movie you like and finds other movies that are mathematically similar to it.

## 🧩 How It Works (The "Secret Sauce")

The computer doesn't understand plots or genres like we do. It only understands numbers. So, we have to turn movie descriptions into numbers. Here is the step-by-step process:

### 1. 🧹 Cleaning the Data (Filtering)
We start with a huge list of movies (`TMDB.csv`).
*   **Problem:** The list has 1 million movies! Many are random home videos, shorts, or documentaries that most people haven't seen.
*   **Solution:** We only keep movies that have **more than 500 votes**. This ensures we only recommend popular, "real" movies.

### 2. 📝 Mixing the Ingredients (Text Processing)
To understand what a movie is about, we combine its most important parts into one long string of text called a "soup" or `encode_list`.
We mix:
*   **Overview:** What happens in the movie.
*   **Genres:** Action, Comedy, Sci-Fi, etc.
*   **Keywords:** Specific tags like "superhero", "space", "magic".
*   **Production Companies:** Marvel, Warner Bros, etc.

**Example for "The Avengers":**
> "superhero alien invasion marvel studios action sci-fi iron man captain america..."

### 3. 🔢 Turning Words into Numbers (Vectorization)
This is the magic part. We use a tool called **TF-IDF** (Term Frequency-Inverse Document Frequency).
*   It counts how many times special words appear.
*   It ignores common words like "the", "and", "is" because they don't help us distinguish movies.
*   It gives higher importance to unique words like "lightsaber" or "wizard".

**Result:** Every movie becomes a list of 5,000 numbers (a "vector").
*   *The Avengers* = `[0.1, 0.0, 0.5, ...]`
*   *Iron Man* = `[0.1, 0.0, 0.4, ...]`
*   *The Notebook* = `[0.0, 0.9, 0.0, ...]`

### 4. 📏 Measuring Similarity (Cosine Similarity)
Now we have numbers, we can measure the "distance" between them.
*   If the numbers are close, the movies are similar.
*   If the numbers are far apart, the movies are different.

When you ask for "The Avengers", the system calculates the distance between "The Avengers" and *every other movie* in the database and gives you the top 10 closest matches.

## 📂 Your Files Explained

### `build_model.py` (The Chef 👨‍🍳)
This is the main script. It does all the hard work described above.
1.  Loads `TMDB.csv`.
2.  Filters out small movies.
3.  Creates the "word soup".
4.  Calculates the numbers (vectors).
5.  Saves the results into two files:
    *   `movies_df.pkl`: The list of movie names.
    *   `feature_array.npz`: The giant list of numbers.

**Run this only once** (or whenever you get new data).

### `recommend.py` (The Waiter 💁)
This script serves the recommendations.
1.  Loads the saved files (`.pkl` and `.npz`).
2.  Asks for a movie name (e.g., "The Avengers").
3.  Finds the similar movies and prints them out.

**Run this whenever you want a recommendation.**

## 🚀 How to Run It

1.  **Train the Model** (Do this first):
    ```bash
    python build_model.py
    ```
    *Wait for it to finish. It creates the "brain" of your system.*

2.  **Get Recommendations**:
    ```bash
    python recommend.py
    ```
    *It will print the top 10 movies similar to "The Avengers".*

---
**Congratulations!** You have built a Content-Based Recommendation System. It recommends items based on their *content* (words/tags) rather than what other users liked.
