import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from thefuzz import process

# Load data
if "data" not in st.session_state:
    df_data = pd.read_csv('data/MAL_data2.csv', index_col=0)
    ratings_df = pd.read_csv('data/users_ratings.csv', index_col=0)
    st.session_state['data'] = df_data
    st.session_state['ratings_df'] = ratings_df

df_data = st.session_state['data']
ratings_df = st.session_state['ratings_df']

# Mapping titles and users to integers
ratings_df['title'] = ratings_df['title'].astype(str)
title_to_int_mapping, int_to_title_mapping = pd.factorize(ratings_df["title"])
ratings_df['titleID'] = title_to_int_mapping

user_to_int_mapping, int_to_user_mapping = pd.factorize(ratings_df["user"])
ratings_df['userID'] = user_to_int_mapping
ratings_df['score'] = ratings_df['score'].astype(int)

# Function to create a pivot table
def create_pivot_table(df):
    pivot_table = df.pivot(index='userID', columns='titleID', values='score').fillna(0)
    return pivot_table

pivot_table = create_pivot_table(ratings_df)

# Function to find similar animes
def find_similar_movies(movie_id, pivot_table, k, p=2):
    movie_vec = pivot_table[movie_id].values.reshape(1, -1)
    kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric='euclidean', p=p)
    kNN.fit(pivot_table.T)
    neighbour = kNN.kneighbors(movie_vec, return_distance=False)
    neighbour_ids = [pivot_table.columns[n] for n in neighbour[0][1:k+1]]
    return neighbour_ids

# Function to find an anime
def anime_finder(title, limit=1):
    all_titles = ratings_df['title'].unique().tolist()
    closest_matches = process.extract(title, all_titles, limit=limit)
    return closest_matches

# Function to get recommendations
def get_content_based_recommendations(title_string, n_recommendations=10):
    title = anime_finder(title_string, limit=1)[0][0]
    movie_id = ratings_df.loc[ratings_df['title'] == title]['titleID'].unique()[0]
    similar_movies = find_similar_movies(movie_id, pivot_table, k=n_recommendations)
    recommendations = [int_to_title_mapping[movie] for movie in similar_movies]
    return recommendations

# Streamlit app layout
st.write("# My Anime List Dashboard ")
st.sidebar.markdown("Developed by [Felipe Gabriel](https://www.linkedin.com/in/felipe-gabriel0/)")

st.markdown("Source: [MAL](https://myanimelist.net/)")

st.markdown("""
The data was obtained by taking the __Top 100__ highest-rated animes according to MAL scores. 
They are calculated as a __weighted average__.
""")

# User input for anime title
anime_title = st.text_input("Enter an anime title you like:")

if anime_title:
    recommendations = get_content_based_recommendations(anime_title)
    st.write("## Recommendations based on your choice:")
    st.write(pd.DataFrame(recommendations, columns=["Recommended Anime"]))
else:
    st.write("Please enter an anime title to get recommendations.")
