from tensorflow import keras
from recommendedsystem.netflix_prize_dataset import Dataset
import os
import tensorflow as tf
import logging

# tf.enable_eager_execution()


class NetflixPrizeModel(keras.Model):
    def __init__(self, movie_count=17771, consumer_count=2649430):
        super(NetflixPrizeModel, self).__init__(name="NetflixPrizeModel")
        self.movie_count = movie_count
        self.consumer_count = consumer_count
        self.embedding_movie = keras.layers.Embedding(
            self.movie_count, 60, input_length=1)
        self.embedding_consumer = keras.layers.Embedding(
            self.consumer_count, 20, input_length=1)
        self.concatenate = keras.layers.Concatenate(axis=2)
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(64)
        self.activation1 = keras.layers.Activation(activation="sigmoid")
        self.dense2 = keras.layers.Dense(64)
        self.activation2 = keras.layers.Activation(activation="sigmoid")
        self.dense3 = keras.layers.Dense(64)
        self.activation3 = keras.layers.Activation(activation="sigmoid")
        self.dense4 = keras.layers.Dense(1)

    def call(self, inputs):
        x_movie = self.flatten(inputs["movie"])
        x_consumer = self.flatten(inputs["consumer"])
        x_movie = self.embedding_movie(x_movie)
        x_consumer = self.embedding_consumer(x_consumer)
        x = self.concatenate([x_movie, x_consumer])
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.activation1(x)
        x = self.dense2(x)
        x = self.activation2(x)
        x = self.dense3(x)
        x = self.activation3(x)
        x = self.dense4(x)
        return x


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
    DATASET_DIR = os.path.expanduser("~/datasets/netflix-prize/tfrecord")
    # CHECKPOINT_PATH = "./checkpoint/model-{epoch:08d}.ckpt"
    CHECKPOINT_PATH = "./checkpoint/model.h5"
    LOG_DIR = "./logs"
    EPOCHS = 1000
    BATCH_SIZE = 60

    model = NetflixPrizeModel()
    model.compile(
        loss="mean_squared_error",
        # loss=loss_fn,
        optimizer='adadelta',
        # optimizer=tf.train.AdamOptimizer(),
        metrics=['accuracy'],
    )
    # model.summary()
    load_model(model, CHECKPOINT_PATH)

    dataset = Dataset(directory=DATASET_DIR)
    dataset_train = dataset.tfdataset("trainingset")

    dataset_evaluate = dataset.tfdataset("qualifyingset")

    def convert(x):
        return {
            "movie": x["movieindex"],
            "consumer": x["consumerindex"]
        }, x["rate"]

    dataset_train = dataset_train.map(convert)
    dataset_evaluate = dataset_evaluate.map(convert)
    dataset_train = dataset_train.batch(BATCH_SIZE)
    dataset_evaluate = dataset_evaluate.batch(1)

    train_model(model, dataset_train, EPOCHS, CHECKPOINT_PATH, LOG_DIR)
    loss, accuracy = evaluate_model(model, dataset_evaluate)
    print("accuracy: {:5.2f}%".format(accuracy * 100))


if __name__ == "__main__":
    main()
