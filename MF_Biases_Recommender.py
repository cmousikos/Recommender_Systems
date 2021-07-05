import tensorflow as tf

# Ignore  the warnings
tf.get_logger().setLevel('ERROR')

import pandas as pd
import numpy as np

from surprise import Dataset
from surprise import Reader

# for neural networks
from keras import backend as K
from keras.layers import Embedding, Input, Flatten,Dropout
from keras.optimizers import Adam
import keras.utils.vis_utils
from keras.utils.vis_utils import plot_model
from keras.layers.merge import dot
import time
import tensorflow as tf
# Ignore  the warnings
tf.get_logger().setLevel('ERROR')
start = time.time()
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(1-K.clip(abs((y_true - y_pred))/5+0.3, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(1-K.clip(abs((y_true - y_pred))/5+0.3, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# import data
df_movies = pd.read_csv('movies.csv', usecols=['movieId', 'title', 'genres'],
                        dtype={'movieId': 'int32', 'title': 'str', 'genres': 'str'})
df_ratings = pd.read_csv('ratings.csv', usecols=['userId', 'movieId', 'rating'],
                         dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

# df_movies.set_index('movieId')
# create categories for unique movieIds
# add a new column for category id
df_ratings.insert(2, "movieId_cat", (df_ratings.movieId.astype('category').cat.codes.values), True)
df_ratings.userId = df_ratings.userId.astype('category').cat.codes.values

users_movies = df_ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)

# create a map dataframe for movieIds
d = {'movieId_cat': df_ratings.movieId_cat.unique(), 'movieId': df_ratings.movieId.unique()}
df_movieId_map = pd.DataFrame(d)

# delete the non-sequential column of movieIds
df_ratings.drop('movieId_cat', axis=1, inplace=True)

P_nap = users_movies.copy()
reader = Reader(rating_scale=(0.5, 5))
data_2 = Dataset.load_from_df(df_ratings, reader)
trainset = data_2.build_full_trainset()
testset = trainset.build_anti_testset(fill=0)

trainset_list = trainset.build_testset()
trainset_df = pd.DataFrame(trainset_list)
trainset_df.columns = ['userId', 'movieId', 'rating']
testset_df = pd.DataFrame(testset)
testset_df.columns = ['userId', 'movieId', 'rating']

# create categories for unique movieIds
trainset_df.userId = trainset_df.userId.astype('category').cat.codes.values
trainset_df.movieId = trainset_df.movieId.astype('category').cat.codes.values

testset_df.userId = testset_df.userId.astype('category').cat.codes.values
testset_df.movieId = testset_df.movieId.astype('category').cat.codes.values

users = trainset_df.userId.unique()
movies = trainset_df.movieId.unique()

# train = df_ratings.copy()

split = np.random.rand(len(trainset_df)) < 0.8
train = trainset_df[split]
valid = trainset_df[~split]

n_movies = len(df_ratings['movieId'].unique())
n_users = len(df_ratings['userId'].unique())
# Input variables
user_input = Input(shape=(1,), dtype='int32', name='user_input')
item_input = Input(shape=(1,), dtype='int32', name='item_input')

MLP_Embedding_User = Embedding(input_dim=n_users, output_dim=64, name='user_embedding')
MLP_Embedding_Item = Embedding(input_dim=n_movies, output_dim=64, name='item_embedding')

# Crucial to flatten an embedding vector!
user_latent = Flatten()(MLP_Embedding_User(user_input))
item_latent = Flatten()(MLP_Embedding_Item(item_input))

user_bias = Embedding(n_users, 1)
movies_bias = Embedding(n_movies,1)

user_bias_vec = Flatten()(user_bias(user_input))
user_bias_vec = Dropout(0.40)(user_bias_vec)
movies_bias_vec = Flatten()(movies_bias(item_input))
movies_bias_vec = Dropout(0.40)(movies_bias_vec)

# Element-wise product of user and item embeddings
# similarity dot product
sim = dot([user_latent, item_latent], name='Simalarity-Dot-Product', axes=1)
#x = Add()([sim,user_bias_vec,movies_bias_vec])

# build model
model = keras.models.Model([user_input, item_input], sim)

model.compile(loss=root_mean_squared_error,optimizer=Adam(lr=1e-4),metrics=[precision_m,'acc',recall_m])


batch_size = 128
epochs = 30

History = model.fit([train.userId, train.movieId], train.rating, batch_size=batch_size,
                    epochs=epochs, validation_data=([valid.userId, valid.movieId], valid.rating),
                    verbose=1)
# predict new ratings for the unrated movies
predictions = model.predict([testset_df.userId, testset_df.movieId])

# convert it to dataframe
df = pd.DataFrame(predictions)
df.columns=['ratings']

# assign the new predicted ratings
testset_df_new = testset_df.copy()
testset_df_new = testset_df_new.assign(rating=df['ratings'])

# re create the new filled user-item matrix only with predicted values
users_movies_predicted = testset_df_new.pivot(index='movieId', columns='userId', values='rating').fillna(0)
new = users_movies_predicted.copy()

end = time.time()
print(end-start)

plot_model(model, to_file="MF-Biases.png", show_shapes=True, show_layer_names=True)

# evaluate the model
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5
import matplotlib.pyplot as plt
plt.plot(History.history['loss'] , 'g')
plt.plot(History.history['val_loss'] , 'b')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()

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
df_movieId_map = df_movieId_map.sort_values(by=['movieId_cat'])



