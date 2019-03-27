from tensorflow import keras
from recommendedsystem.netflix_prize_dataset import Dataset
import os
import tensorflow as tf
import logging

# tf.enable_eager_execution()


def create_model():
    MOVIE_COUNT = 17771
    CONSUMER_COUNT = 2649430

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
    def loss_fn(y_true, y_pred):
        print(input_movie)
        return y_true - y_pred
    model.compile(
        loss="mean_squared_error",
        # loss=loss_fn,
        optimizer='adadelta',
        metrics=['accuracy'],
    )
    return model


def load_model(model, checkpoint):
    latest_checkpoint = None
    if os.path.exists(checkpoint):
        latest_checkpoint = checkpoint
    else:
        latest_checkpoint = tf.train.latest_checkpoint(
            os.path.dirname(checkpoint))
    if latest_checkpoint:
        logging.info("load model from {}".format(latest_checkpoint))
        model.load_weights(latest_checkpoint)


def train_model(model, dataset, epochs, checkpoint, logdir):
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint,
        save_weights_only=True,
        verbose=1,
    )
    tensorboard_callback = keras.callbacks.TensorBoard(logdir)
    model.fit(
        dataset,
        epochs=epochs,
        steps_per_epoch=24,
        callbacks=[checkpoint_callback, tensorboard_callback],
    )
    # model.save_weights(MODEL_PATH)


def evaluate_model(model, dataset):
    loss, accuracy = model.evaluate(dataset, steps=1)
    return loss, accuracy


def main():
    DATASET_DIR = os.path.expanduser("~/repository/datasets/netflix-prize/tfrecord")
    # CHECKPOINT_PATH = "./checkpoint/model-{epoch:08d}.ckpt"
    CHECKPOINT_PATH = "./checkpoint/model.h5"
    LOG_DIR= "./logs"
    EPOCHS = 1000
    BATCH_SIZE = 60

    model = create_model()
    model.summary()
    load_model(model, CHECKPOINT_PATH)

    dataset = Dataset(directory=DATASET_DIR)
    dataset_train = dataset.tfdataset("trainingset")
    dataset_train = dataset_train.batch(BATCH_SIZE)

    dataset_evaluate = dataset.tfdataset("qualifyingset")
    dataset_evaluate = dataset_evaluate.batch(1)

    def convert(x):
        return {
            "movieindex": x["movieindex"],
            "consumerindex": x["consumerindex"]
        }, x["rate"]

    dataset_train = dataset_train.map(convert)
    dataset_evaluate = dataset_evaluate.map(convert)

    train_model(model, dataset_train, EPOCHS, CHECKPOINT_PATH, LOG_DIR)
    loss, accuracy = evaluate_model(model, dataset_evaluate)
    print("accuracy: {:5.2f}%".format(accuracy * 100))


if __name__ == "__main__":
    main()
