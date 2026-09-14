"""Evaluation logic for the DigitCNN model."""

import torch
import torch.nn as nn

from src.data import get_dataloaders
from src.train import get_device, train_model


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate the model on a dataset and return the average loss and accuracy."""
    model.eval()  # set the model to evaluation mode

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # disable gradient computation
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total
    return average_loss, accuracy

def get_predictions(model, dataloader, device):
    """Run the model over a dataset and return the true labels and predictions."""
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu()

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.tolist())

    return all_labels, all_predictions

if __name__ == "__main__":
    from src.data import get_datasets
    from src.utils import plot_confusion_matrix, plot_predictions, plot_incorrect_predictions

    device = get_device()
    model, _, _ = train_model()

    _, test_loader = get_dataloaders()
    _, test_dataset = get_datasets()
    criterion = nn.CrossEntropyLoss()

    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    true_labels, predicted_labels = get_predictions(model, test_loader, device)
    plot_confusion_matrix(true_labels, predicted_labels)

    plot_predictions(model, test_dataset, device)
    plot_incorrect_predictions(model, test_dataset, device)