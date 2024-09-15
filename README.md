# khan
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

# Step 1: Load the datasets
movies = pd.read_csv('movies.csv')  # contains movieId, title, genres
ratings = pd.read_csv('ratings.csv')  # contains userId, movieId, rating, timestamp

# Step 2: Merge datasets
movie_data = pd.merge(ratings, movies, on='movieId')

# Step 3: Create a user-movie matrix
user_movie_matrix = movie_data.pivot_table(index='userId', columns='title', values='rating')

# Step 4: Fill NaN with 0 (optional, but some algorithms require this)
user_movie_matrix.fillna(0, inplace=True)

# Step 5: Compute the cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Step 6: Recommendation function
def recommend_movies(user_id, num_recommendations=5):
    # Get the similarity scores for the user
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    
    # Get the movies that the similar users liked
    similar_users_ratings = user_movie_matrix.loc[similar_users.index]
    
    # Compute the weighted sum of ratings from similar users
    weighted_ratings = similar_users_ratings.T.dot(similar_users) / similar_users.sum()
    
    # Find the movies the target user hasn't rated
    user_ratings = user_movie_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings == 0]
    
    # Filter out movies that the user has already rated
    recommendations = weighted_ratings.loc[unrated_movies.index].sort_values(ascending=False)
    
    return recommendations.head(num_recommendations)

# Step 7: Test the recommendation system
user_id = 1  # For example, we want recommendations for user with ID 1
recommendations = recommend_movies(user_id)
print(f"Recommendations for user {user_id}:\n", recommendations)

