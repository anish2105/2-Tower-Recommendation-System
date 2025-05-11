import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os

df_movies = pd.read_csv(r'D:\Projects\ML\RecSys\ml-32m\movies.csv')
df_ratings = pd.read_csv(r'D:\Projects\ML\RecSys\ml-32m\ratings.csv')
df_links = pd.read_csv(r'D:\Projects\ML\RecSys\ml-32m\links.csv')
df_tags = pd.read_csv(r'D:\Projects\ML\RecSys\ml-32m\tags.csv')

print("Length of tags",len(df_tags))
print("Length of movies",len(df_movies))
print("Length of ratings",len(df_ratings))
print("Length of links",len(df_links))

merged_link_movie_df = pd.merge(df_movies, df_links, on='movieId', how='inner')
merged_link_movie_df.head(5)
print(merged_link_movie_df)
print(len(merged_link_movie_df))

merged_link_movie_tags_df = pd.merge(merged_link_movie_df, df_tags, on='movieId', how='inner')
merged_link_movie_tags_df.head(5)
print(merged_link_movie_tags_df)
print(len(merged_link_movie_tags_df))

df_ratings = df_ratings
df_ratings = df_ratings[['movieId' , 'userId' , 'rating']]
print(merged_link_movie_tags_df)
print(len(df_ratings))
print("-"*50)
merged_df = pd.merge(merged_link_movie_tags_df, df_ratings, on=['movieId', 'userId'], how='inner')

merged_df['genres'] = merged_df['genres'].str.replace('|', ', ')
merged_df['genre_tag'] = merged_df['genres'] + ', ' + merged_df['tag']

print(merged_df.head(5))
print(len(merged_df))

merged_df.to_csv(r"D:\Projects\ML\RecSys\data\merged_movie_df.csv", index = False)