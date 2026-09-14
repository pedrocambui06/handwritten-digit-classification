"""Dataset exploration and visualization utilities."""

import torch
import matplotlib.pyplot as plt

from src.data import get_datasets
from sklearn.metrics import confusion_matrix


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

def plot_confusion_matrix(true_labels, predicted_labels, save_path: str = "results/figures/confusion_matrix.png") -> None:
    """Compute and plot the confusion matrix as a heatmap."""
    cm = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.colorbar()

    # Write the count inside each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    plt.savefig(save_path)
    print(f"Saved confusion matrix to {save_path}")


def plot_predictions(model, dataset, device, num_samples: int = 10, save_path: str = "results/figures/predictions.png") -> None:
    """Plot a grid of test images with their actual and predicted labels."""
    model.eval()
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))

    with torch.no_grad():
        for i, ax in enumerate(axes.flat):
            image, label = dataset[i]
            output = model(image.unsqueeze(0).to(device))
            prediction = output.argmax(dim=1).item()

            ax.imshow(image.squeeze(), cmap="gray")
            ax.set_title(f"Actual: {label}\nPredicted: {prediction}")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved predictions to {save_path}")


def plot_incorrect_predictions(model, dataset, device, num_samples: int = 10, save_path: str = "results/figures/incorrect_predictions.png") -> None:
    """Find and plot examples where the model predicted the wrong digit."""
    model.eval()
    incorrect_examples = []

    with torch.no_grad():
        for image, label in dataset:
            output = model(image.unsqueeze(0).to(device))
            prediction = output.argmax(dim=1).item()

            if prediction != label:
                incorrect_examples.append((image, label, prediction))
            if len(incorrect_examples) >= num_samples:
                break

    if not incorrect_examples:
        print("No incorrect predictions found among the samples checked.")
        return

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for ax, (image, label, prediction) in zip(axes.flat, incorrect_examples):
        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(f"Actual: {label}\nPredicted: {prediction}")
        ax.axis("off")

    for ax in axes.flat[len(incorrect_examples):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved incorrect predictions to {save_path}")

if __name__ == "__main__":
    train_dataset, test_dataset = get_datasets()

    print_dataset_info(train_dataset, test_dataset)
    plot_sample_images(train_dataset)