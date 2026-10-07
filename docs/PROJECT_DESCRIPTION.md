# Project Description

## Title
Indian Sign Language Static Sign Translator Using Computer Vision and Deep Learning

## Problem statement
Communication barriers can arise when people who do not understand Indian Sign Language interact with people who use it. This project explores a computer-vision approach for recognizing static ISL signs and converting them into readable labels.

## Objective
Develop a lightweight prototype that:
1. Detects one or two hands from a camera feed.
2. Extracts hand landmarks.
3. Normalizes the landmarks.
4. Classifies the resulting feature vector using a neural network.
5. Displays the predicted ISL class and confidence in real time.

## Technologies
- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy
- scikit-learn
- Matplotlib / Seaborn
- Jupyter / Google Colab

## Key technical idea
Instead of feeding full camera images directly to the neural network, the project first converts each detected hand into 21 three-dimensional landmarks. This produces a compact 126-feature representation for two hands.

## Output
The live application displays the recognized static sign, for example:

`A (94%)`

when the model's predicted class is A and the prediction probability exceeds the configured confidence threshold.
