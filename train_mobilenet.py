from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# -----------------------------
# SETTINGS
# -----------------------------

TRAIN_DIR = Path("dataset_train_augmented")
VAL_DIR = Path("dataset_split/val")
TEST_DIR = Path("dataset_split/test")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

# Apple Silicon GPU if available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# LOAD DATA
# -----------------------------

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

print("Classes:", train_dataset.classes)

print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# -----------------------------
# LOAD PRETRAINED MOBILENETV2
# -----------------------------

print("\nLoading pretrained MobileNetV2...")

weights = models.MobileNet_V2_Weights.DEFAULT

model = models.mobilenet_v2(
    weights=weights
)


# Freeze pretrained convolution layers
for parameter in model.features.parameters():
    parameter.requires_grad = False


# Replace final classifier for 2 classes
model.classifier[1] = nn.Linear(
    model.last_channel,
    2
)

model = model.to(device)


# -----------------------------
# LOSS + OPTIMIZER
# -----------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# TRAINING
# -----------------------------

train_losses = []
val_accuracies = []

print("\nTraining started...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)

    train_losses.append(average_loss)


    # -------------------------
    # VALIDATION
    # -------------------------

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    val_accuracy = correct / total

    val_accuracies.append(val_accuracy)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.4f}"
    )


# -----------------------------
# TEST
# -----------------------------

print("\nTesting model...")

model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)


print("\n=== MOBILENETV2 TEST RESULTS ===")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes,
        zero_division=0
    )
)


# -----------------------------
# CONFUSION MATRIX
# -----------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=test_dataset.classes
)

display.plot()

plt.title(
    "MobileNetV2 Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "mobilenet_confusion_matrix.png",
    dpi=300
)

plt.show()


# -----------------------------
# TRAINING LOSS GRAPH
# -----------------------------

plt.figure()

plt.plot(
    range(1, EPOCHS + 1),
    train_losses,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "MobileNetV2 Training Loss"
)

plt.grid()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "mobilenet_training_loss.png",
    dpi=300
)

plt.show()


# -----------------------------
# VALIDATION ACCURACY GRAPH
# -----------------------------

plt.figure()

plt.plot(
    range(1, EPOCHS + 1),
    val_accuracies,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")

plt.title(
    "MobileNetV2 Validation Accuracy"
)

plt.grid()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "mobilenet_validation_accuracy.png",
    dpi=300
)

plt.show()


# -----------------------------
# SAVE RESULTS
# -----------------------------

with open(
    RESULTS_DIR / "mobilenet_results.txt",
    "w"
) as file:

    file.write(
        "MobileNetV2 Transfer Learning Results\n"
    )

    file.write(
        "=====================================\n"
    )

    file.write(
        f"Test Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Precision: {precision:.4f}\n"
    )

    file.write(
        f"Recall: {recall:.4f}\n"
    )

    file.write(
        f"F1-score: {f1:.4f}\n"
    )


# Save model
torch.save(
    model.state_dict(),
    RESULTS_DIR / "mobilenet_model.pth"
)

print("\nModel and results saved in results/")
