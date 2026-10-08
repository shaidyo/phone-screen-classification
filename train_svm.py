from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


IMAGE_SIZE = (64, 64)

TRAIN_DIR = Path("dataset_train_augmented")
VAL_DIR = Path("dataset_split/val")
TEST_DIR = Path("dataset_split/test")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CLASSES = ["laptop_closed", "laptop_open"]


def load_dataset(directory):
    X = []
    y = []

    for label, class_name in enumerate(CLASSES):
        class_dir = directory / class_name

        files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))

        for file in files:
            img = Image.open(file).convert("L")
            img = img.resize(IMAGE_SIZE)

            array = np.array(img, dtype=np.float32) / 255.0

            X.append(array.flatten())
            y.append(label)

    return np.array(X), np.array(y)


print("Loading datasets...")

X_train, y_train = load_dataset(TRAIN_DIR)
X_val, y_val = load_dataset(VAL_DIR)
X_test, y_test = load_dataset(TEST_DIR)

print()
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print()
print("Training SVM...")


model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=10,
        gamma="scale"
    ))
])

model.fit(X_train, y_train)

print("Training complete.")


# Validation
val_predictions = model.predict(X_val)

val_accuracy = accuracy_score(y_val, val_predictions)

print()
print("Validation Accuracy:", round(val_accuracy, 4))


# Test
test_predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(y_test, test_predictions, zero_division=0)
recall = recall_score(y_test, test_predictions, zero_division=0)
f1 = f1_score(y_test, test_predictions, zero_division=0)

print()
print("=== SVM TEST RESULTS ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=CLASSES,
        zero_division=0
    )
)


# Confusion Matrix
cm = confusion_matrix(y_test, test_predictions)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASSES
)

display.plot()

plt.title("SVM Confusion Matrix")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "svm_confusion_matrix.png",
    dpi=300
)

plt.show()


# Save metrics
with open(RESULTS_DIR / "svm_results.txt", "w") as f:
    f.write("SVM Baseline Results\n")
    f.write("====================\n")
    f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
    f.write(f"Test Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n")

print()
print("Results saved in results/")
