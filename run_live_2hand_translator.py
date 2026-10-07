"""Real-time Indian Sign Language static sign translator.

Run:
    python run_live_2hand_translator.py
Press Q to quit.
"""
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "isl_static_2hand_model.keras"
LABELS_PATH = ROOT / "models" / "isl_static_2hand_labels.npy"

CONFIDENCE_THRESHOLD = 0.80
CAMERA_INDEX = 0


def extract_2hand_landmarks(image, hands_model):
    """Return 126 normalized features in [Left hand, Right hand] order."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_model.process(image_rgb)

    features = np.zeros(126, dtype=np.float32)
    if not results.multi_hand_landmarks:
        return features, results

    for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
        try:
            hand_label = results.multi_handedness[i].classification[0].label
        except (IndexError, AttributeError):
            continue

        wrist = hand_landmarks.landmark[0]
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend((lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z))
        flat = np.asarray(coords, dtype=np.float32)

        if hand_label == "Left":
            features[:63] = flat
        elif hand_label == "Right":
            features[63:] = flat

    return features, results


def main():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError(
            "Model files not found. Expected:\n"
            f"  {MODEL_PATH}\n  {LABELS_PATH}"
        )

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = np.load(LABELS_PATH, allow_pickle=True)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions.")

    print("Starting webcam. Press Q to quit.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands_model:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            features, results = extract_2hand_landmarks(frame, hands_model)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            prediction = model.predict(features[np.newaxis, :], verbose=0)[0]
            confidence = float(np.max(prediction))
            index = int(np.argmax(prediction))

            if results.multi_hand_landmarks and confidence >= CONFIDENCE_THRESHOLD:
                text = f"{labels[index]} ({confidence * 100:.0f}%)"
            elif results.multi_hand_landmarks:
                text = "Low confidence"
            else:
                text = "No hand detected"

            cv2.rectangle(frame, (0, 0), (500, 70), (0, 0, 0), -1)
            cv2.putText(
                frame, text, (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2
            )
            cv2.imshow("Indian Sign Language Translator", frame)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Translator stopped.")


if __name__ == "__main__":
    main()
