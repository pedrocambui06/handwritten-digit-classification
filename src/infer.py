"""Model saving, loading, and single-image inference."""

import os

import torch
import torch.nn.functional as F

from src.data import get_datasets
from src.model import DigitCNN
from src.train import get_device

MODEL_PATH = "models/cnn_model.pth"


def save_model(model, path: str = MODEL_PATH) -> None:
    """Save the model's learned parameters (state_dict) to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path: str = MODEL_PATH, device=None):
    """Create a new DigitCNN and load previously saved parameters into it."""
    device = device or get_device()
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def predict_single_image(model, image, device):
    """Run inference on a single image and return the predicted class and its confidence."""
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        probabilities = F.softmax(output, dim=1)
        confidence, prediction = probabilities.max(dim=1)

    return prediction.item(), confidence.item()


if __name__ == "__main__":
    device = get_device()
    model = load_model(device=device)

    _, test_dataset = get_datasets()
    image, actual_label = test_dataset[0]

    predicted_label, confidence = predict_single_image(model, image, device)

    print(f"Actual digit: {actual_label}")
    print(f"Predicted digit: {predicted_label}")
    print(f"Confidence: {confidence * 100:.2f}%")