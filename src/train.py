"""Training loop for the DigitCNN model."""

import torch
import torch.nn as nn
import torch.optim as optim

from src.data import get_dataloaders
from src.model import DigitCNN

LEARNING_RATE = 0.001
NUM_EPOCHS = 5


def get_device() -> torch.device:
    """Return the GPU device if available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Run a single training epoch and return the average loss and accuracy."""
    model.train()  # set the model to training mode

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # clear gradients from the previous step
        outputs = model(images)        # forward pass
        loss = criterion(outputs, labels)  # compute the loss
        loss.backward()                # backpropagation: compute gradients
        optimizer.step()               # update model weights

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total
    return average_loss, accuracy


def train_model():
    """Train the DigitCNN model for NUM_EPOCHS epochs and return the trained model and metrics."""
    device = get_device()
    print(f"Using device: {device}")

    train_loader, _ = get_dataloaders()

    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    train_accuracies = []

    for epoch in range(1, NUM_EPOCHS + 1):
        loss, accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(loss)
        train_accuracies.append(accuracy)

        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {accuracy * 100:.2f}%")

    return model, train_losses, train_accuracies


if __name__ == "__main__":
    from src.utils import plot_training_curves

    model, train_losses, train_accuracies = train_model()
    plot_training_curves(train_losses, train_accuracies)