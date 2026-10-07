"""Extract MediaPipe hand landmarks from an ISL image dataset.

Expected dataset layout:
data/ISL_Static_Data/
    A/*.jpg
    B/*.jpg
    ...
    0/*.jpg
    ...
"""

from pathlib import Path
import argparse
import cv2
import numpy as np
import mediapipe as mp


def extract_2hand_landmarks(image, hands_model):
    features = np.zeros(126, dtype=np.float32)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_model.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
        if i >= len(results.multi_handedness):
            continue
        label = results.multi_handedness[i].classification[0].label
        wrist = hand_landmarks.landmark[0]
        values = []
        for lm in hand_landmarks.landmark:
            values.extend((lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z))
        values = np.asarray(values, dtype=np.float32)

        if label == "Left":
            features[:63] = values
        elif label == "Right":
            features[63:] = values

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ISL_Static_Data")
    parser.add_argument("--output", default="data/ISL_Static_2Hand_Features")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png"}
    mp_hands = mp.solutions.hands
    total, skipped = 0, 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    ) as hands_model:
        for class_dir in sorted(p for p in data_path.iterdir() if p.is_dir()):
            out_dir = output_path / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in extensions:
                    continue
                image = cv2.imread(str(image_path))
                if image is None:
                    skipped += 1
                    continue

                features = extract_2hand_landmarks(image, hands_model)
                if features is None:
                    skipped += 1
                    continue

                np.save(out_dir / f"{image_path.stem}.npy", features)
                total += 1

    print(f"Saved features for {total} images; skipped {skipped}.")


if __name__ == "__main__":
    main()
