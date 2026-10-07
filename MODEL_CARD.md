# Model Card — ISL Static 2-Hand Classifier

## Overview
This project uses MediaPipe Hands to convert a camera image into 126 normalized landmark features:
2 hands × 21 landmarks × (x, y, z).

A fully connected neural network classifies the resulting feature vector into 36 static classes:
digits 0–9 and letters A–Z.

## Intended use
- Educational demonstrations
- Academic projects
- Prototype real-time ISL recognition
- Computer-vision experimentation

## Limitations
- The model is a static-sign classifier, not a full continuous sentence translator.
- Recognition quality depends on lighting, camera angle, hand orientation, distance, and the training dataset.
- The model was trained on landmark features; it does not directly learn from raw RGB pixels.
- Only the classes represented in the supplied label file are supported.
- Confidence scores are model probabilities and should not be interpreted as guaranteed correctness.

## Architecture
Input: 126 features

Dense(128, ReLU)
→ Dropout(0.30)
→ Dense(256, ReLU)
→ Dropout(0.40)
→ Dense(128, ReLU)
→ Dense(36, Softmax)

## Preprocessing
For each detected hand, landmark coordinates are translated relative to that hand's wrist. Left-hand landmarks occupy features 0–62 and right-hand landmarks occupy features 63–125.

## Training split
The original training workflow uses an 80/20 stratified train/test split with random_state=42.

## Classes
0–9 and A–Z.
