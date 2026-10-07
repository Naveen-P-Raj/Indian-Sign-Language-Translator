"""Evaluate the trained 2-hand ISL classifier."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


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
    return np.asarray(X, dtype=np.float32), labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/ISL_Static_2Hand_Features")
    parser.add_argument("--model", default="models/isl_static_2hand_model.keras")
    parser.add_argument("--labels", default="models/isl_static_2hand_labels.npy")
    parser.add_argument("--output", default="results/confusion_matrix.png")
    args = parser.parse_args()

    X, labels = load_features(args.features)
    class_names = np.load(args.labels, allow_pickle=True)
    mapping = {name: i for i, name in enumerate(class_names)}
    y_ids = np.asarray([mapping[x] for x in labels])

    _, X_test, _, y_test = train_test_split(
        X, y_ids, test_size=0.20, random_state=42, stratify=y_ids
    )

    model = tf.keras.models.load_model(args.model)
    pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print(classification_report(
        y_test, pred, labels=np.arange(len(class_names)),
        target_names=class_names, zero_division=0
    ))

    cm = confusion_matrix(y_test, pred, labels=np.arange(len(class_names)))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.title("Indian Sign Language - 2-Hand Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
