"""Train an MLP on MNIST. Requires TensorFlow and an MNIST download.

Unlike the CNN example, this model flattens each image before learning:
each hidden neuron connects to every pixel. The CNN instead applies shared
filters to local image regions, using spatial structure to detect features.
Both models output probabilities for the same ten digit classes.
"""

from tensorflow import keras


def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train[0:100]
    print(x_train.shape)
    print(x_test.shape)
    # exit(-1)

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.summary()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
