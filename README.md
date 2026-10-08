# Laptop State Classification Using Computer Vision

This project classifies laptops into two classes:

- laptop_open
- laptop_closed

## Dataset

The dataset was self-collected using a smartphone camera.

Original dataset:
- 31 laptop_open images
- 31 laptop_closed images

The training subset was augmented using:
- rotation
- brightness adjustment
- contrast adjustment
- horizontal flip
- blur
- crop/zoom

## Models

### Baseline
Support Vector Machine (SVM)

Test Accuracy: 66.67%

### Improved Model
MobileNetV2 with Transfer Learning

Test Accuracy: 91.67%

## Evaluation Metrics

The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## MobileNetV2 Results

- Accuracy: 91.67%
- Precision: 100%
- Recall: 83.33%
- F1-score: 90.91%
