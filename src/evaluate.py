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


if __name__ == "__main__":
    device = get_device()

    # Train the model (model saving/loading will be added in a later commit)
    model, _, _ = train_model()

    _, test_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()

    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")