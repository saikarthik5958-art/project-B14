
# transfer_learning_domain_adaptive.py
# Design of a Transfer Learning-Based Deep Learning Model for Domain-Adaptive Classification
# Domain: Deep Learning

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# 1. Load Real-World Dataset (CIFAR-10)
# -----------------------------
def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Use subset for faster training
    X_train = X_train[:10000]
    y_train = y_train[:10000]
    X_test = X_test[:2000]
    y_test = y_test[:2000]

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return X_train, X_test, y_train, y_test


# -----------------------------
# 2. Transfer Learning Model
# -----------------------------
def build_model():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(32, 32, 3)
    )

    base_model.trainable = False  # freeze base model

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    output = Dense(10, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------
# 3. Training
# -----------------------------
def train_model(model, X_train, y_train, X_test, y_test):
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    model.save("transfer_learning_model.h5")
    print("Transfer learning model saved")

    return history


# -----------------------------
# 4. Evaluation
# -----------------------------
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))


# -----------------------------
# 5. Training Visualization
# -----------------------------
def plot_history(history):
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Transfer Learning Training Performance")
    plt.savefig("training_accuracy.png")
    plt.show()


# -----------------------------
# 6. Main Execution
# -----------------------------
def main():
    X_train, X_test, y_train, y_test = load_data()

    model = build_model()
    history = train_model(model, X_train, y_train, X_test, y_test)
    evaluate_model(model, X_test, y_test)
    plot_history(history)


if __name__ == "__main__":
    main()
