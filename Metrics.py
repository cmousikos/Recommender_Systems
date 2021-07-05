import pandas as pd
import numpy as np
from scipy import spatial
from Two_tower_SVD_NN_Recommender import show
# from Memory_based_Recommender import show
# from MF_Biases_Recommender import show
# from MF_NN_Recommender import show
# from NN_Recommender import show
# from MF_NN_Two_Model_Hybrid_Recommender import show
# from SVD_Netflix_Recommender import show
# from SVD_item_item_Recommender import show
# from KNN_Recommender import show


# import data
df_movies = pd.read_csv('movies.csv', usecols=['movieId', 'title', 'genres'],
                        dtype={'movieId': 'int32', 'title': 'str', 'genres': 'str'})
df_ratings = pd.read_csv('ratings.csv', usecols=['userId', 'movieId', 'rating'],
                         dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

# add a new column for category id
df_ratings.insert(2, "movieId_cat", (df_ratings.movieId.astype('category').cat.codes.values), True)
df_ratings.userId = df_ratings.userId.astype('category').cat.codes.values

users_movies = df_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# create a map dataframe for movieIds
d = {'movieId_cat': df_ratings.movieId_cat.unique(), 'movieId': df_ratings.movieId.unique()}
df_movieId_map = pd.DataFrame(d)

# delete the non-sequential column of movieIds
df_ratings.drop('movieId_cat', axis=1, inplace=True)
# genres = np.empty(len(df_ratings), dtype=object)

# for i in range(len(df_ratings)):
#    for j in range(len(df_movies)):
#        if df_ratings.movieId[i]==df_movies.movieId[j]:
#            genres[i] = df_movies.genres[j]
#            continue

# df_ratings.insert(3,"genres",genres,True)

df_ratings_genres = pd.read_csv('ratings_genres.csv', usecols=['userId', 'movieId', 'rating', 'genres'],
                                dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32', 'genres': 'str'})


# find percentage of genres in each user
# init variables
def Percentages_ratings(target_user):
    Adventure = 0
    Comedy = 0
    Drama = 0
    Action = 0
    Romance = 0
    Mystery = 0
    Crime = 0
    War = 0
    Animation = 0
    Children = 0
    Thriller = 0
    Sci_Fi = 0
    Western = 0
    Fantasy = 0
    Musical = 0
    OverAll = 0
    for i in range(len(df_ratings_genres)):
        if df_ratings_genres.userId[i] == target_user:
            if df_ratings_genres.genres[i].__contains__('Adventure'):
                Adventure = Adventure + 1
                OverAll = OverAll + 1
            if df_ratings_genres.genres[i].__contains__('Comedy'):
                Comedy = Comedy + 1
                OverAll = OverAll + 1
            if df_ratings_genres.genres[i].__contains__('Action'):
                Action = Action + 1
                OverAll = OverAll + 1
            if df_ratings_genres.genres[i].__contains__('Romance'):
                Romance = Romance + 1
                OverAll = OverAll + 1
            if df_ratings_genres.genres[i].__contains__('Mystery'):
                Mystery = Mystery + 1
                OverAll = OverAll + 1
            if df_ratings_genres.genres[i].__contains__('Crime'):
                Crime = Crime + 1
                OverAll = OverAll + 1
            if df_ratings_genres.genres[i].__contains__('War'):
                War = War + 1
                OverAll = OverAll + 1
            if df_ratings_genres.genres[i].__contains__('Animation'):
                Animation = Animation + 1
                OverAll = OverAll + 1
            if df_ratings_genres.genres[i].__contains__('Children'):
                Children = Children + 1
                OverAll = OverAll + 1
            if df_ratings_genres.genres[i].__contains__('Thriller'):
                Thriller = Thriller + 1
                OverAll = OverAll + 1
            if df_ratings_genres.genres[i].__contains__('Sci_Fi'):
                Sci_Fi = Sci_Fi + 1
                OverAll = OverAll + 1
            if df_ratings_genres.genres[i].__contains__('Western'):
                Western = Western + 1
                OverAll = OverAll + 1
            if df_ratings_genres.genres[i].__contains__('Fantasy'):
                Fantasy = Fantasy + 1
                OverAll = OverAll + 1
            if df_ratings_genres.genres[i].__contains__('Musical'):
                Musical = Musical + 1
                OverAll = OverAll + 1
            if df_ratings_genres.genres[i].__contains__('Drama'):
                Drama = Drama + 1
                OverAll = OverAll + 1
    Adventure_p = Adventure / OverAll
    Drama_p = Drama / OverAll
    Comedy_p = Comedy / OverAll
    Action_p = Action / OverAll
    Romance_p = Romance / OverAll
    Mystery_p = Mystery / OverAll
    Crime_p = Crime / OverAll
    War_p = War / OverAll
    Animation_p = Animation / OverAll
    Children_p = Children / OverAll
    Thriller_p = Thriller / OverAll
    Sci_Fi_p = Sci_Fi / OverAll
    Western_p = Western / OverAll
    Fantasy_p = Fantasy / OverAll
    Musical_p = Musical / OverAll
    data = {'Adventure': [Adventure_p], 'Drama': [Drama_p], 'Comedy': [Comedy_p], 'Action': [Action_p],
            'Romance': [Romance_p], 'Mystery': [Mystery_p], 'Crime': [Crime_p], 'War': [War_p],
            'Animation': [Animation_p], 'Children': [Children_p], 'Thriller': [Thriller_p], 'Sci_Fi': [Sci_Fi_p],
            'Western': [Western_p], 'Fantasy': [Fantasy_p], 'Musical': [Musical_p]}
    ind = {'Percentages'}
    df = pd.DataFrame(data, index=ind)
    return df


def Percentages_recommendations(target_user):
    recommendations = show(target_user)
    Adventure = 0
    Drama = 0
    Comedy = 0
    Action = 0
    Romance = 0
    Mystery = 0
    Crime = 0
    War = 0
    Animation = 0
    Children = 0
    Thriller = 0
    Sci_Fi = 0
    Western = 0
    Fantasy = 0
    Musical = 0
    OverAll = 0
    for i in range(len(recommendations)):
        if recommendations[i][1].__contains__('Adventure'):
            Adventure = Adventure + 1
            OverAll = OverAll + 1
        if recommendations[i][1].__contains__('Comedy'):
            Comedy = Comedy + 1
            OverAll = OverAll + 1
        if recommendations[i][1].__contains__('Action'):
            Action = Action + 1
            OverAll = OverAll + 1
        if recommendations[i][1].__contains__('Romance'):
            Romance = Romance + 1
            OverAll = OverAll + 1
        if recommendations[i][1].__contains__('Mystery'):
            Mystery = Mystery + 1
            OverAll = OverAll + 1
        if recommendations[i][1].__contains__('Crime'):
            Crime = Crime + 1
            OverAll = OverAll + 1
        if recommendations[i][1].__contains__('War'):
            War = War + 1
            OverAll = OverAll + 1
        if recommendations[i][1].__contains__('Animation'):
            Animation = Animation + 1
            OverAll = OverAll + 1
        if recommendations[i][1].__contains__('Children'):
            Children = Children + 1
            OverAll = OverAll + 1
        if recommendations[i][1].__contains__('Thriller'):
            Thriller = Thriller + 1
            OverAll = OverAll + 1
        if recommendations[i][1].__contains__('Sci_Fi'):
            Sci_Fi = Sci_Fi + 1
            OverAll = OverAll + 1
        if recommendations[i][1].__contains__('Western'):
            Western = Western + 1
            OverAll = OverAll + 1
        if recommendations[i][1].__contains__('Fantasy'):
            Fantasy = Fantasy + 1
            OverAll = OverAll + 1
        if recommendations[i][1].__contains__('Musical'):
            Musical = Musical + 1
            OverAll = OverAll + 1
        if recommendations[i][1].__contains__('Drama'):
            Drama = Drama + 1
            OverAll = OverAll + 1

    Adventure_p = Adventure / OverAll
    Drama_p = Drama / OverAll
    Comedy_p = Comedy / OverAll
    Action_p = Action / OverAll
    Romance_p = Romance / OverAll
    Mystery_p = Mystery / OverAll
    Crime_p = Crime / OverAll
    War_p = War / OverAll
    Animation_p = Animation / OverAll
    Children_p = Children / OverAll
    Thriller_p = Thriller / OverAll
    Sci_Fi_p = Sci_Fi / OverAll
    Western_p = Western / OverAll
    Fantasy_p = Fantasy / OverAll
    Musical_p = Musical / OverAll
    data = {'Adventure': [Adventure_p], 'Drama': [Drama_p], 'Comedy': [Comedy_p], 'Action': [Action_p],
            'Romance': [Romance_p], 'Mystery': [Mystery_p], 'Crime': [Crime_p], 'War': [War_p],
            'Animation': [Animation_p], 'Children': [Children_p], 'Thriller': [Thriller_p], 'Sci_Fi': [Sci_Fi_p],
            'Western': [Western_p], 'Fantasy': [Fantasy_p], 'Musical': [Musical_p]}
    ind = {'Percentages'}
    df = pd.DataFrame(data, index=ind)
    return df


# similarity between the percentages of genres
def Personalization(target_user):
    df_1 = Percentages_ratings(target_user)
    df_2 = Percentages_recommendations(target_user)
    vector_1 = df_1.to_numpy()
    vector_2 = df_2.to_numpy()
    cosine_similarity = 1 - spatial.distance.cosine(vector_1, vector_2)
    return cosine_similarity


similarities_matrix = np.empty(users_movies.shape[0])
for i in range(users_movies.shape[0]):
    similarities_matrix[i] = Personalization(i)
    mean_pers = similarities_matrix.mean()



def Similarity(target_user):
    i = 0
    s = 0
    d = 0
    recommended = show(target_user)
    recommended_movieIds = recommended[:, 2]
    n = len(recommended_movieIds)
    while i < len(recommended_movieIds):
        j = i
        while j < len(recommended_movieIds):
            cosine_similarity = 1 - spatial.distance.cosine(users_movies[:][recommended_movieIds[i]],
                                                            users_movies[:][recommended_movieIds[j]])
            s = s + cosine_similarity
            d = d + (1 - cosine_similarity)
            j = j + 1
        i = i + 1
    S = s / ((n / 2) * (n - 1))
    D = d / ((n / 2) * (n - 1))
    return D


Diversity_matrix = np.empty(users_movies.shape[0])
for i in range(users_movies.shape[0]):
    Diversity_matrix[i] = Similarity(i)
    mean_div = Diversity_matrix.mean()


def All_recommendations():
    recommendations_df = pd.DataFrame(index=range(users_movies.shape[0] * 10),
                                      columns=['UserId', 'Title', 'Genre', 'MovieId'])
    start = 0
    step = 10
    for i in range(users_movies.shape[0]):
        end = start + step
        recommendations = show(i)
        recommendations_df.UserId[start:end] = i
        recommendations_df.Title[start:end] = recommendations[:, 0]
        recommendations_df.Genre[start:end] = recommendations[:, 1]
        recommendations_df.MovieId[start:end] = recommendations[:, 2]
        start = start + step

    return recommendations_df


df = All_recommendations()
dups_movieId = df.pivot_table(index=['MovieId'], aggfunc='size')


def Novelty(target_user):
    k = 0
    l = 0
    idx = np.empty(10)
    nov = np.empty(10)
    n = len(df.UserId.unique())
    for i in range(len(df)):
        if df.UserId[i] == target_user:
            idx[k] = i
            k = k +1
    for j in idx:
        count = dups_movieId[df.MovieId[j]]
        nov[l] = 1 - (count/n)
        l = l + 1
    mean_nov = nov.mean()
    return mean_nov

Novelty_matrix = np.empty(users_movies.shape[0])
for i in range(users_movies.shape[0]):
    Novelty_matrix[i] = Novelty(i)
    mean_nov = Novelty_matrix.mean()
