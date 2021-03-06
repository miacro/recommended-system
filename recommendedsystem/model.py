from tensorflow import keras
from recommendedsystem.netflix_prize_dataset import Dataset
import os
import tensorflow as tf
import logging
import math

# tf.enable_eager_execution()


def create_netflixprize_model(movie_count=17771, consumer_count=2649430):
    input_movie = keras.layers.Input(name="movie", shape=(1, ))
    input_consumer = keras.layers.Input(name="consumer", shape=(1, ))
    x_movie = keras.layers.Embedding(
        movie_count, 60, input_length=1)(input_movie)
    x_consumer = keras.layers.Embedding(
        consumer_count, 20, input_length=1)(input_consumer)
    x = keras.layers.Concatenate()([x_movie, x_consumer])
    x = keras.layers.Flatten()(x)

    def add_layer(x, units=64):
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation="relu")(x)
        x = keras.layers.Dense(
            units,
            kernel_initializer=keras.initializers.RandomUniform(),
            bias_initializer=keras.initializers.Zeros(),
            kernel_regularizer=keras.regularizers.l2(0.01),
            bias_regularizer=keras.regularizers.l2(0.01))(x)
        return x

    x = add_layer(x, 64)
    x = add_layer(x, 64)
    x = add_layer(x, 64)
    x = add_layer(x, 1)
    return keras.Model(inputs=[input_movie, input_consumer], outputs=x)


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


def train_model(model, dataset, validation, epochs, checkpoint, logdir):
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint,
        save_weights_only=True,
        verbose=1,
    )
    tensorboard_callback = keras.callbacks.TensorBoard(logdir)
    model.fit(
        dataset,
        epochs=epochs,
        validation_data=validation,
        steps_per_epoch=100,
        validation_steps=20,
        callbacks=[checkpoint_callback, tensorboard_callback],
    )
    # model.save_weights(MODEL_PATH)


def evaluate_model(model, dataset):
    return model.evaluate(dataset, steps=10000)


def main():
    DATASET_DIR = os.path.expanduser("~/datasets/netflix-prize/tfrecord")
    # CHECKPOINT_PATH = "./checkpoint/model-{epoch:08d}.ckpt"
    CHECKPOINT_PATH = os.path.expanduser("./checkpoint/model.h5")
    LOG_DIR = os.path.expanduser("./logs")
    EPOCHS = 40
    BATCH_SIZE = 20000
    BUFFER_SIZE = 40000
    if not os.path.isdir(os.path.dirname(CHECKPOINT_PATH)):
        os.makedirs(os.path.dirname(CHECKPOINT_PATH))

    model = create_netflixprize_model()

    def loss_fn(y_true, y_pred):
        # return keras.backend.sqrt(
        #    keras.metrics.mean_squared_error(y_true, y_pred * 5))
        return keras.metrics.mean_squared_error(y_true, y_pred * 5)

    model.compile(
        loss=loss_fn,
        # loss="mean_squared_error",
        optimizer='adadelta',
        # optimizer=tf.train.AdamOptimizer(),
    )
    # model.summary()
    load_model(model, CHECKPOINT_PATH)

    dataset = Dataset(directory=DATASET_DIR)
    dataset_train = dataset.tfdataset("trainingset")
    dataset_evaluate = dataset.tfdataset("qualifyingset")
    dataset_validate = dataset.tfdataset("probeset")
    dataset_train = dataset_train.shuffle(buffer_size=BUFFER_SIZE)
    dataset_evaluate = dataset_evaluate.shuffle(buffer_size=BUFFER_SIZE)
    dataset_validate = dataset_validate.shuffle(buffer_size=BUFFER_SIZE)
    dataset_train = dataset_train.repeat(3)

    def convert(x):
        return {
            "movie": x["movieindex"],
            "consumer": x["consumerindex"]
        }, x["rate"]

    dataset_train = dataset_train.map(convert)
    dataset_evaluate = dataset_evaluate.map(convert)
    dataset_validate = dataset_validate.map(convert)
    dataset_train = dataset_train.batch(BATCH_SIZE)
    dataset_evaluate = dataset_evaluate.batch(20)
    dataset_validate = dataset_validate.batch(60)
    train_model(model, dataset_train, dataset_validate, EPOCHS,
                CHECKPOINT_PATH, LOG_DIR)
    loss = evaluate_model(model, dataset_evaluate)
    print("rmse: {:5.2f}".format(math.sqrt(loss)))


if __name__ == "__main__":
    main()
