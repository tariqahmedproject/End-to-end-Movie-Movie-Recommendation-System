from flask import Flask, render_template, request
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the preprocessed movie data
movies_df = pd.read_csv('Content_Base_Recommendation_System.csv')

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit the vectorizer on the combined 'genres' and 'keywords' columns
vectorizer.fit(movies_df['genres'] + ' ' + movies_df['keywords'])

# Function to get recommendations
def get_recommendations(title):
    # Get the index of the movie
    try:
        movie_index = movies_df[movies_df['title'] == title].index[0]
    except:
        return []  # Return an empty list if the movie is not found

    # Get the TF-IDF representation of the movie
    movie_vector = vectorizer.transform([movies_df.iloc[movie_index]['genres'] + ' ' + movies_df.iloc[movie_index]['keywords']])

    # Calculate cosine similarity between the movie and all other movies
    cosine_similarities = cosine_similarity(movie_vector, vectorizer.transform(movies_df['genres'] + ' ' + movies_df['keywords']))

    # Get the indices of the most similar movies
    most_similar_indices = cosine_similarities.argsort()[0][-5:][::-1]

    # Return the titles of the most similar movies
    return movies_df.iloc[most_similar_indices]['title'].tolist()

# Initialize Flask app
app = Flask(__name__)

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for getting recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    # Get the movie title from the form
    movie_title = request.form['movie_title']

    # Get recommendations
    recommendations = get_recommendations(movie_title)

    # Render the results page
    return render_template('results.html', movie_title=movie_title, recommendations=recommendations)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)