from tensorflow import keras

if __name__ == "__main__":
    movie_count = 17771
    user_count = 2649430
    model_left = keras.Sequential()
    model_left.add(keras.layers.Embedding(movie_count, 60, input_length=1))
    model_right = keras.Sequential()
    model_right.add(keras.layers.Embedding(user_count, 20, input_length=1))
    model = keras.Sequential()
    model.add(keras.Merge()[model_left, model_right], mode="concat")
    model.add(keras.Flatten())
    model.add(keras.Dense(64))
    model.add(keras.activations.sigmoid())
    model.add(keras.layers.Dense(64))
    model.add(keras.activations.sigmoid())
    model.add(keras.layers.Dense(64))
    model.add(keras.activations.sigmoid())
    model.add(keras.layers.Dense(1))
    model.compile(loss="mean_seqared_error", optimizer='adadelta')
