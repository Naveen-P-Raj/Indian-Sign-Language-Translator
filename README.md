# 🇮🇳 Indian Sign Language Static Sign Translator

A real-time **Indian Sign Language (ISL) static sign recognition** project using **MediaPipe Hands, TensorFlow/Keras, OpenCV, NumPy and scikit-learn**.

The system detects up to two hands from a webcam frame, extracts 126 normalized hand-landmark features, and uses a trained neural network to recognize **36 static classes: digits 0–9 and letters A–Z**.

> **Note:** This is a static sign classifier/prototype. It is not a complete continuous ISL sentence translator.

## ✨ Features

- Real-time webcam recognition
- Two-hand detection
- MediaPipe 21-point hand landmarks
- 126-dimensional normalized feature vector
- TensorFlow/Keras neural-network classifier
- 36 classes: `0–9` and `A–Z`
- Confidence threshold for predictions
- Training notebook for Google Colab
- Reproducible feature-extraction, training and evaluation scripts
- Saved trained model and label mapping

## 🧠 How it works

```text
Webcam / Image
      ↓
OpenCV
      ↓
MediaPipe Hands
      ↓
Detect up to 2 hands
      ↓
21 landmarks × 3 coordinates × 2 hands
      ↓
126 normalized features
      ↓
Dense Neural Network
      ↓
Softmax prediction
      ↓
ISL sign + confidence
```

### Feature layout

| Hand | Features |
|---|---:|
| Left | 0–62 |
| Right | 63–125 |
| Total | 126 |

Each hand is normalized relative to its wrist, making the representation less sensitive to the hand's absolute position in the camera frame.

## 📁 Repository structure

```text
Indian-Sign-Language/
├── README.md
├── MODEL_CARD.md
├── LICENSE
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── run_live_2hand_translator.py
├── models/
│   ├── isl_static_2hand_model.keras
│   └── isl_static_2hand_labels.npy
├── notebooks/
│   └── ISL_Static_model.ipynb
├── src/
│   ├── extract_features.py
│   ├── train_model.py
│   └── evaluate_model.py
├── data/
│   └── README.md
├── results/
└── docs/
```

## 🚀 Quick start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Indian-Sign-Language.git
cd Indian-Sign-Language
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the trained model

The trained model is already included in `models/`.

```bash
python run_live_2hand_translator.py
```

Allow webcam access when prompted.

Press **Q** to stop.

## 🏋️ Train from your own dataset

Place your image dataset under:

```text
data/ISL_Static_Data/
├── 0/
├── 1/
...
├── A/
├── B/
...
└── Z/
```

Then extract MediaPipe features:

```bash
python src/extract_features.py
```

Train the neural network:

```bash
python src/train_model.py
```

Optional training controls:

```bash
python src/train_model.py --epochs 100 --batch-size 32
```

## 📈 Reported results

The supplied training notebook reports the following evaluation setup and results:

| Metric | Reported value |
|---|---:|
| Total feature samples | 22,854 |
| Classes | 36 |
| Training samples | 18,283 |
| Test samples | 4,571 |
| Test accuracy | ≈99% |
| Model parameters | 86,820 |

The reported ≈99% accuracy comes from an 80/20 stratified split of the same feature dataset using `random_state=42`. It should therefore be interpreted as a held-out split result, not as an independent real-world benchmark.

The class-level evaluation in the supplied notebook shows precision, recall, and F1 scores around 0.90–1.00 across the 36 classes, with the lowest reported class-level F1 values for `N` (0.94) and `M` (0.95).

## 📊 Evaluate the model

After feature extraction and training:

```bash
python src/evaluate_model.py
```

This prints a classification report and saves:

```text
results/confusion_matrix.png
```

## 📓 Google Colab notebook

`notebooks/ISL_Static_model.ipynb` contains the original training/evaluation workflow used for the project.

The notebook was originally designed around Google Drive paths. For a clean local GitHub workflow, the `src/` scripts are recommended.

## 🗂️ Dataset

The image dataset is intentionally **not included** in this repository because datasets may have separate licensing and redistribution requirements.

See [`data/README.md`](data/README.md) for the required folder structure.

## 🔬 Model

The classifier uses:

```text
Input: 126
↓
Dense: 128, ReLU
↓
Dropout: 0.30
↓
Dense: 256, ReLU
↓
Dropout: 0.40
↓
Dense: 128, ReLU
↓
Dense: 36, Softmax
```

The saved model is:

`models/isl_static_2hand_model.keras`

The class mapping is:

`models/isl_static_2hand_labels.npy`

## ⚠️ Limitations

- Static signs only
- Does not model temporal hand movement
- Does not generate complete sentences
- Performance depends on dataset quality and diversity
- Webcam lighting/background can affect detection
- Signs outside the 36 trained classes are not supported
- A high confidence score does not guarantee a correct prediction

## 🔮 Future improvements

- Add dynamic ISL gesture recognition using sequences/LSTM/Transformer models
- Add sentence formation and language processing
- Add text-to-speech output
- Add more diverse training images
- Add data augmentation
- Add a browser/mobile interface
- Add real-time prediction smoothing
- Evaluate on an independent test dataset
- Package as a Streamlit application

## 👨‍💻 Author

**Naveen P Raj**  
MS Artificial Intelligence and Data Science

GitHub: `https://github.com/Naveen-P-Raj`

## 📄 License

Released under the MIT License. See `LICENSE`.
