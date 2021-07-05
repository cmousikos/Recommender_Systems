import time

import numpy as np
import pandas as pd
from numpy import diag
from numpy import zeros
from scipy import spatial
from scipy.linalg import svd

start = time.time()
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
mat_movies_users = users_movies.to_numpy()

# preprocess user_movies matrix in order to eliminate all missing data values
# Compute the average of all rows and the average of all columns
# rows are movies and columns are users
movies_average = mat_movies_users.mean(axis=1)
user_average = mat_movies_users.mean(axis=0)

users_movies_filled = users_movies.copy()
mat_users_movies_filled = users_movies_filled.to_numpy()

# Replace all matrix entries that have no values
# with the corresponding column average
for movieId in range(users_movies.shape[0] - 1):
    for userId in range(users_movies.shape[1] - 1):
        if users_movies.values[movieId][userId] == 0:
            mat_users_movies_filled[movieId][userId] = movies_average[movieId]

users_movies_norm = users_movies_filled.copy()
mat_users_movies_norm = users_movies_norm.to_numpy()
# Subtract the corresponding row average
# from all the slots of the new filled-in matrix
for movieId in range(users_movies.shape[0] - 1):
    for userId in range(users_movies.shape[1] - 1):
        mat_users_movies_norm[movieId][userId] = mat_users_movies_filled[movieId][userId] - user_average[userId]


# using numpy
# Compute the SVD of the normalized matrix and obtain matrices U, S and VT
a = mat_users_movies_norm
U, S, VT = svd(a)
Sigma = zeros((a.shape[0], a.shape[1]))
Sigma[:a.shape[1], :a.shape[1]] = diag(S)
B = U.dot(Sigma.dot(VT))



# Perform the dimensionality reduction step by
# keeping only k diagonal entries from matrix S to
# obtain a k × k matrix, Sk. Similarly, matrices Uk
# and Vk of size m×k and k×n are generated
embeddings = 10
n = VT.shape[0]
m = U.shape[0]
Sk = Sigma[:embeddings, :embeddings]
VTk = VT[:embeddings, :n]
Uk = U[:m, :embeddings]

# the ”reduced” user-item matrix, user_movies_red, is obtained by
# user_movies_red = Uk· Sk· VTk while  denotes the rating
# by user ui on item ij as included in this reduced matrix.

# Compute √Sk and then calculate two matrix products: Uk·√SkT,
# which represents m users and
# √Sk·VTk , which represents n items in the k dimensional
# feature space. We are particularly interested
# in the latter matrix, of size k ×n, whose entries
# represent the ”meta” ratings provided by the
# k pseudo-users on the n items. A ”meta” rating
# assigned by pseudo-user ui on item [i][j] is denoted
# by mr[i][j].

sqrt_Sk = np.sqrt(Sk)
sqrt_Sk_T = np.transpose(sqrt_Sk)
meta_movies = np.matmul(Uk, sqrt_Sk_T)
meta_user = np.matmul(sqrt_Sk,VTk)

P_nap = pd.DataFrame(meta_movies)

# compute similarity between the items of the df P_nap
# Similarity between lines
target_movie = 0
w = len(P_nap.index)
cosine_similarity = np.empty(shape=(w,w)) #compute only the diagonal
while target_movie < len(P_nap.index):
    similarMovie = 0
    temp = similarMovie + target_movie
    while similarMovie < temp+1:
        cosine_similarity[target_movie][similarMovie] = 1 - spatial.distance.cosine(P_nap.values[target_movie], P_nap.values[similarMovie])
        cosine_similarity[similarMovie][target_movie] = cosine_similarity[target_movie][similarMovie] #fill the up triangle
        similarMovie = similarMovie + 1
    target_movie = target_movie + 1
arr = np.array(cosine_similarity)
df = pd.DataFrame(data=arr[0:,0:])
top_similarities_matrix = df.abs().values.argsort(1)[:, -11:][:, ::-1]
top_similarities_matrix = top_similarities_matrix[:,1:11]

P_nap_final = users_movies.copy()
def recommender(array):
    temp = 0
    i = 0
    k = 0
    j = 0
    user_id = 0
    movie_id = 0
    numerator = 0
    temp_2 = 0

    # fill the new matrix with new predicted values for each user
    while movie_id < len(array.index):

        user_id = 0
        while user_id < len(array.columns):

            k = 0
            temp = 0
            temp_2 = 0

            if array.values[movie_id][user_id] == 0:
                for l in range(10):
                    idx = top_similarities_matrix[movie_id][k]
                    numerator = (cosine_similarity[idx][movie_id]) * ((array.values[idx][user_id]) + user_average[user_id])
                    temp = temp + numerator
                    temp_2 = temp_2 + abs(cosine_similarity[movie_id][k])
                    k = k + 1
            if (temp > 0) & (temp_2 > 0):
                P_nap_final.values[movie_id][user_id] = temp / temp_2
            user_id = user_id + 1
        movie_id = movie_id + 1
    return P_nap_final


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
P_nap_final = recommender(users_movies)
new = recommend_10_highest(P_nap_final)
end = time.time()
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
