import time

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

start = time.time()
movies = 'movies.csv'
ratings = 'ratings.csv'

# import data
df_movies = pd.read_csv('movies.csv', usecols=['movieId', 'title', 'genres'],
                        dtype={'movieId': 'int32', 'title': 'str', 'genres': 'str'})
df_ratings = pd.read_csv('ratings.csv', usecols=['userId', 'movieId', 'rating'],
                         dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
df_movies.set_index('movieId')
# create categories for unique movieIds
# add a new column for category id
df_ratings.insert(2, "movieId_cat", (df_ratings.movieId.astype('category').cat.codes.values), True)
df_ratings.userId = df_ratings.userId.astype('category').cat.codes.values

users_movies = df_ratings.pivot(index='movieId_cat', columns='userId', values='rating').fillna(0)

# create a map dataframe for movieIds
d = {'movieId_cat': df_ratings.movieId_cat.unique(), 'movieId': df_ratings.movieId.unique()}
df_movieId_map = pd.DataFrame(d)

# delete the non-sequential column of movieIds
df_ratings.drop('movieId', axis=1, inplace=True)

P_nap = users_movies.copy()

model_knn = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=10)  # train algorithm without zeros
model_knn.fit(users_movies)



# users_movies.to_excel('output.xlsx')

def recommender(array, model, neighbors):
    temp = 0
    i = 0
    k = 0
    j = 0
    user_id = 0
    movie_id = 0
    numerator = 0
    temp_2 = 0
    distances, indices = model.kneighbors()
    similarity = 1 - distances

    # fill the new matrix with new predicted values for each user
    while movie_id < len(array.index):

        user_id = 0
        while user_id < len(array.columns):

            k = 0
            temp = 0
            temp_2 = 0

            if array.values[movie_id][user_id] == 0:
                for l in range(neighbors):
                    idx = indices[movie_id][k]
                    numerator = (similarity[movie_id][k]) * (array.values[idx][user_id])
                    temp = temp + numerator
                    temp_2 = temp_2 + similarity[movie_id][k]
                    k = k + 1
            if (temp > 0) & (temp_2 > 0):
                P_nap.values[movie_id][user_id] = temp / temp_2
            user_id = user_id + 1
        movie_id = movie_id + 1
    return P_nap,similarity,indices


def recommend_10_highest(array):
    movie_id = 0
    new =  array.copy()
    # Compare the matrices between them so to get only the new
    while movie_id < len(users_movies.index):
        user_id = 0
        while user_id < len(users_movies.columns):
            if users_movies.values[movie_id][user_id] == array.values[movie_id][user_id]:
                new.values[movie_id][user_id] = 0
            user_id = user_id + 1
        movie_id = movie_id + 1
    # Recommend the 10 with highest ratings
    return new

P_nap, similiraty, indices = recommender(users_movies, model_knn, 10)
new = recommend_10_highest(P_nap)
end = time.time()
total_time = end - start
print(end - start)


def show(target_user_id):
    movies_ids_cat = new.nlargest(10, [target_user_id])
    movieId = np.empty(10, dtype=int)
    recommendations = np.empty([10,3],dtype=object)
    k = 0
    for i in range(len(movies_ids_cat)):
        for j in range(len(df_movieId_map)):
            if movies_ids_cat.index[i] == df_movieId_map.movieId_cat[j]:
                movieId[k] = df_movieId_map.movieId[j]
                k = k + 1
    for k in range(10):
        for l in range(len(df_movies)):
            if df_movies.movieId[l] == movieId[k]:
                recommendations[k][0] = df_movies['title'][l]
                recommendations[k][1] = df_movies['genres'][l]
                recommendations[k][2] = df_movies['movieId'][l]
                #print(df_movies['title'][l])
                #print(df_movies['genres'][l])
    return recommendations

