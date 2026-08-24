"""
Complete CNN Training Example with TensorBoard
Dataset: CIFAR-10 (10 classes, 32x32 color images)
Classes: airplane, automobile, bird, cat, deer,
         dog, frog, horse, ship, truck
"""

import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint


# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─── Config ───────────────────────────────────────────────────────────────────
NUM_CLASSES = 10
IMAGE_SIZE = (32, 32, 3)
BATCH_SIZE = 64
EPOCHS = 30
LOG_DIR = os.path.join("logs", "fit",
                       datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ─── 1. Load & preprocess data ────────────────────────────────────────────────
def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Use a small subset to keep training fast
    x_train, y_train = x_train[:10_000], y_train[:10_000]
    x_test,  y_test = x_test[:2_000],  y_test[:2_000]

    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test,  NUM_CLASSES)

    print(f"Train samples : {len(x_train)}")
    print(f"Test  samples : {len(x_test)}")
    print(f"Input shape   : {x_train.shape[1:]}")
    return (x_train, y_train), (x_test, y_test)


# ─── 2. Data augmentation ─────────────────────────────────────────────────────
def build_augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="augmentation")


# ─── 3. Build CNN model ───────────────────────────────────────────────────────
def build_model():
    inputs = keras.Input(shape=IMAGE_SIZE, name="input_image")

    # Augmentation (only active during training)
    x = build_augmentation()(inputs, training=True)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu",
                      name="conv5")(x)
    x = layers.BatchNormalization(name="bn5")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Dropout(0.25, name="drop3")(x)

    # Classifier head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.5, name="drop4")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = models.Model(inputs, outputs, name="cifar10_cnn")
    return model


# ─── 4. TensorBoard callbacks ─────────────────────────────────────────────────
def build_callbacks():
    tb_callback = TensorBoard(
        log_dir=LOG_DIR,
        histogram_freq=1,          # Log weight histograms every epoch
        write_graph=True,       # Visualize the model graph
        write_images=True,       # Log model weights as images
        update_freq="epoch",
        profile_batch=0,          # Disable profiling for speed
    )

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint = ModelCheckpoint(
        filepath=os.path.join("checkpoints", "best_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    return [tb_callback, early_stop, checkpoint]


# ─── 5. Custom TensorBoard: log sample images with labels ─────────────────────
def log_sample_images(x_test, y_test):
    """Write a grid of test images with true/predicted labels to TensorBoard."""
    file_writer = tf.summary.create_file_writer(
        os.path.join(LOG_DIR, "images"))

    # Select one image per class
    indices = []
    for c in range(NUM_CLASSES):
        idx = np.where(np.argmax(y_test, axis=1) == c)[0][0]
        indices.append(idx)

    sample_images = x_test[indices]                   # (10, 32, 32, 3)
    # (1, 10, 32, 32, 3) — not valid for TB
    sample_images = np.expand_dims(sample_images, 0)

    # TensorBoard expects (batch, H, W, C); log each image individually
    with file_writer.as_default():
        for i, idx in enumerate(indices):
            img = x_test[idx:idx+1]           # (1, 32, 32, 3)
            tf.summary.image(
                name=f"sample/{CLASS_NAMES[i]}",
                data=img,
                step=0,
            )
    file_writer.flush()
    print(f"Sample images logged to TensorBoard at: {LOG_DIR}/images")


# ─── 6. Confusion-matrix image callback ───────────────────────────────────────
class ConfusionMatrixCallback(keras.callbacks.Callback):
    """Logs a confusion-matrix image to TensorBoard at the end of each epoch."""

    def __init__(self, x_val, y_val, log_dir):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val          # one-hot
        self.writer = tf.summary.create_file_writer(
            os.path.join(log_dir, "confusion_matrix"))

    def _plot_confusion_matrix(self, cm):
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(NUM_CLASSES),
               yticks=np.arange(NUM_CLASSES),
               xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
               title="Confusion Matrix",
               ylabel="True label", xlabel="Predicted label")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img = tf.image.decode_png(buf.getvalue(), channels=4)
        return tf.expand_dims(img, 0)   # (1, H, W, 4)

    def on_epoch_end(self, epoch, logs=None):
        from sklearn.metrics import confusion_matrix

        y_pred = np.argmax(self.model.predict(self.x_val, verbose=0), axis=1)
        y_true = np.argmax(self.y_val, axis=1)
        cm = confusion_matrix(y_true, y_pred)
        cm_img = self._plot_confusion_matrix(cm)

        with self.writer.as_default():
            tf.summary.image("confusion_matrix", cm_img, step=epoch)
        self.writer.flush()


# ─── 7. Main ──────────────────────────────────────────────────────────────────
def main():
    os.makedirs("checkpoints", exist_ok=True)

    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()

    # Log sample images before training
    log_sample_images(x_test, y_test)

    # Build model
    model = build_model()
    model.summary()

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    callbacks = build_callbacks()
    callbacks.append(
        ConfusionMatrixCallback(x_test, y_test, LOG_DIR)
    )

    # Train
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Final evaluation
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest loss     : {loss:.4f}")
    print(f"Test accuracy : {accuracy * 100:.2f}%")

    # ── How to view TensorBoard ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("To open TensorBoard, run in a terminal:")
    print(f"  tensorboard --logdir {os.path.abspath('logs')}")
    print("Then open  http://localhost:6006  in your browser.")
    print("=" * 60)

    return history


if __name__ == "__main__":
    main()
