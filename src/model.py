"""CNN architecture for MNIST digit classification."""

import torch.nn as nn


class DigitCNN(nn.Module):
    """A simple Convolutional Neural Network for classifying MNIST digits."""

    def __init__(self):
        super().__init__()

        # First convolutional block: 1 input channel (grayscale) -> 16 filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # 28x28 -> 14x14

        # Second convolutional block: 16 filters -> 32 filters
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # 14x14 -> 7x7

        # Flatten the [32, 7, 7] feature maps into a single vector of size 32*7*7
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # 10 outputs, one per digit class

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    import torch

    model = DigitCNN()

    # Simulate a batch of 4 grayscale 28x28 images
    dummy_input = torch.randn(4, 1, 28, 28)
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")   # expected: [4, 10]