import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
import pickle

# Read the dataset
df = pd.read_csv('data/sample30.csv')

# Split into train and test
train, test = train_test_split(df, train_size=0.70, random_state=45)

# Create user-item rating matrix
train_pivot = pd.pivot_table(index='reviews_username',
                            columns='name',
                            values='reviews_rating',
                            data=train).fillna(1)

# Cosine similarity function
def cosine_similarity(df):
    mean_df = np.nanmean(df, axis=1)
    substracted_df = (df.T - mean_df).T
    user_correlation = 1 - pairwise_distances(substracted_df.fillna(0), metric='cosine')
    user_correlation[np.isnan(user_correlation)] = 0
    return user_correlation, substracted_df

# Calculate user similarity
user_corr_matrix, normalized_df = cosine_similarity(train_pivot)
user_corr_matrix[user_corr_matrix < 0] = 0

# Generate predicted ratings
user_pred_ratings = np.dot(user_corr_matrix, train_pivot.fillna(0))

# Final ratings matrix
user_final_rating = np.multiply(user_pred_ratings, train_pivot)

# Save matrix
pickle.dump(user_final_rating, open('pickle_files/user_final_rating.pkl', 'wb'))
print("Generated and saved user_final_rating.pkl")