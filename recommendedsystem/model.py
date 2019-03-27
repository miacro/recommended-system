from tensorflow import keras
from recommendedsystem.netflix_prize_dataset import Dataset
import os
import tensorflow as tf
import logging
# tf.enable_eager_execution()

if __name__ == "__main__":
    MOVIE_COUNT = 17771
    CONSUMER_COUNT = 2649430
    DATASET_DIR = os.path.expanduser("~/datasets/netflix-prize/tfrecord")
    # CHECKPOINT_PATH = "./checkpoint/model-{epoch:08d}.ckpt"
    CHECKPOINT_PATH = "./checkpoint/model.h5"
    MODEL_PATH = "./model/model.h5"

    EPOCHS = 3
    BATCH_SIZE = 60

    input_movie = keras.layers.Input(name="movieindex", shape=(1, ))
    input_consumer = keras.layers.Input(name="consumerindex", shape=(1, ))
    x_movie = keras.layers.Embedding(
        MOVIE_COUNT, 60, input_length=1)(input_movie)
    x_consumer = keras.layers.Embedding(
        CONSUMER_COUNT, 20, input_length=1)(input_consumer)
    x = keras.layers.Concatenate()([x_movie, x_consumer])
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64)(x)
    x = keras.layers.Activation(activation="sigmoid")(x)
    x = keras.layers.Dense(64)(x)
    x = keras.layers.Activation(activation="sigmoid")(x)
    x = keras.layers.Dense(64)(x)
    x = keras.layers.Activation(activation="sigmoid")(x)
    y = keras.layers.Dense(1)(x)
    model = keras.models.Model([input_movie, input_consumer], y)
    model.compile(loss="mean_squared_error", optimizer='adadelta')

    dataset = Dataset(directory=DATASET_DIR)
    dataset = dataset.tfdataset("trainingset")
    dataset = dataset.batch(BATCH_SIZE)

    def convert(x):
        return {
            "movieindex": x["movieindex"],
            "consumerindex": x["consumerindex"]
        }, x["rate"]

    dataset = dataset.map(convert)

    model.summary()
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        CHECKPOINT_PATH,
        save_weights_only=True,
        verbose=1,
    )
    latest_checkpoint = None
    if os.path.exists(CHECKPOINT_PATH):
        latest_checkpoint = CHECKPOINT_PATH
    else:
        latest_checkpoint = tf.train.latest_checkpoint(
            os.path.dirname(CHECKPOINT_PATH))
    if latest_checkpoint:
        logging.info("load model from {}".format(latest_checkpoint))
        model.load_weights(latest_checkpoint)
    model.fit(
        dataset,
        epochs=EPOCHS,
        steps_per_epoch=24,
        callbacks=[checkpoint_callback],
    )
    # model.save_weights(MODEL_PATH)
