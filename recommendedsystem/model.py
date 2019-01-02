from tensorflow import keras
from . import netflix_prize_dataset as Dataset

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

    epochs = 10000
    batch_size = 60

    dataset = Dataset(directory="~/datasets/netflix-prize/tfrecord")
    dataset = dataset.tfdataset("trainingset")
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    feautres = iterator.get_next()
    model.fit(
        [feautres["movieindex"], feautres["consumerindex"]],
        labels=feautres["rate"],
        epochs=epochs,
        batch_size=batch_size)
