"""Train the 2-hand static ISL classifier from extracted 126-D features."""
from pathlib import Path
import argparse
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_features(feature_dir):
    X, labels = [], []
    for class_dir in sorted(Path(feature_dir).iterdir()):
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.glob("*.npy")):
            data = np.load(path)
            if data.shape == (126,):
                X.append(data)
                labels.append(class_dir.name)
    if not X:
        raise RuntimeError("No valid 126-feature files found.")
    return np.asarray(X, dtype=np.float32), labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/ISL_Static_2Hand_Features")
    parser.add_argument("--model", default="models/isl_static_2hand_model.keras")
    parser.add_argument("--labels", default="models/isl_static_2hand_labels.npy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    X, labels = load_features(args.features)
    encoder = LabelEncoder()
    y_ids = encoder.fit_transform(labels)
    y = to_categorical(y_ids)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y_ids
    )

    model = Sequential([
        Input(shape=(126,)),
        Dense(128, activation="relu"),
        Dropout(0.30),
        Dense(256, activation="relu"),
        Dropout(0.40),
        Dense(128, activation="relu"),
        Dense(len(encoder.classes_), activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.labels).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model)
    np.save(args.labels, encoder.classes_)
    print("Saved model:", args.model)
    print("Saved labels:", args.labels)


if __name__ == "__main__":
    main()
