"""Dataset exploration and visualization utilities."""

import matplotlib.pyplot as plt

from src.data import get_datasets


def print_dataset_info(train_dataset, test_dataset) -> None:
    """Print basic statistics about the MNIST dataset."""
    image, label = train_dataset[0]

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Image tensor shape: {image.shape}")
    print(f"Image data type: {image.dtype}")
    print(f"Number of classes: {len(train_dataset.classes)}")


def plot_sample_images(dataset, num_samples: int = 10, save_path: str = "results/figures/sample_images.png") -> None:
    """Plot a grid of sample images with their labels and save it to disk."""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    for i, ax in enumerate(axes.flat):
        image, label = dataset[i]
        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved sample images to {save_path}")

def plot_training_curves(train_losses, train_accuracies,
                          loss_save_path: str = "results/figures/training_loss.png",
                          accuracy_save_path: str = "results/figures/training_accuracy.png") -> None:
    """Plot and save training loss and accuracy curves over epochs."""
    epochs = range(1, len(train_losses) + 1)

    # Loss curve
    plt.figure()
    plt.plot(epochs, train_losses, marker="o")
    plt.title("Training Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(loss_save_path)
    print(f"Saved training loss curve to {loss_save_path}")

    # Accuracy curve
    plt.figure()
    plt.plot(epochs, train_accuracies, marker="o")
    plt.title("Training Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(accuracy_save_path)
    print(f"Saved training accuracy curve to {accuracy_save_path}")

if __name__ == "__main__":
    train_dataset, test_dataset = get_datasets()

    print_dataset_info(train_dataset, test_dataset)
    plot_sample_images(train_dataset)