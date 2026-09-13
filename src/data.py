"""MNIST dataset loading and DataLoader creation."""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import BATCH_SIZE, DATA_DIR

# MNIST-specific normalization values (precomputed mean and std of the dataset)
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def get_transform() -> transforms.Compose:
    """Return the preprocessing pipeline applied to every MNIST image.

    Steps:
    1. Convert the PIL image into a PyTorch tensor with values in [0, 1].
    2. Normalize the tensor using MNIST's known mean and standard deviation,
       centering the pixel values around 0.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])


def get_datasets():
    """Download (if needed) and return the MNIST train and test datasets."""
    transform = get_transform()

    train_dataset = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )

    return train_dataset, test_dataset


def get_dataloaders():
    """Return the training and test DataLoaders, ready for iteration."""
    train_dataset, test_dataset = get_datasets()

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # Quick manual check: confirm the data loads and has the expected shape
    train_loader, test_loader = get_dataloaders()

    images, labels = next(iter(train_loader))
    print(f"Batch of images shape: {images.shape}")   # [batch_size, 1, 28, 28]
    print(f"Batch of labels shape: {labels.shape}")   # [batch_size]
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")