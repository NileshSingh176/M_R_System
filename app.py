import streamlit as st
import pickle
import joblib
import pandas as pd
import numpy as np

# Load movie data
with open("Movies.pkl", "rb") as file:
    movies_data = pickle.load(file)

# Load similarity matrix
# Make sure you saved it with joblib.dump(similarity, 'similarity.pkl', compress=3)
similarity = joblib.load("similarity.pkl")

# Get movie titles
movies_list = movies_data["title"].values

# Recommendation function
def recommend(movie):
    try:
        # Normalize movie input
        movie = movie.lower().strip()
        movie_index = movies_data[movies_data["title"].str.lower().str.strip() == movie].index[0]
        distances = similarity[movie_index]

        # Get top 5 similar movies (excluding the selected movie itself)
        sorted_result = sorted(enumerate(distances), key=lambda x: x[1], reverse=True)[1:6]

        recommended_movies = [movies_data.iloc[i[0]]["title"] for i in sorted_result]
        return recommended_movies
    except IndexError:
        return ["Movie not found in dataset."]
    except Exception as e:
        return [f"Error: {str(e)}"]

# Streamlit UI
st.title("🎬 Movie Recommender System")

selected_movie = st.selectbox("Select a movie:", movies_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    for movie in recommendations:
        st.write("👉", movie)