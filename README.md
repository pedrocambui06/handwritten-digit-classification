# Handwritten Digit Classification with a Convolutional Neural Network (PyTorch)

A step-by-step, reproducible implementation of an image classification pipeline using PyTorch, built as a personal learning project on Deep Learning fundamentals. This repository documents the full workflow, from raw MNIST data to a trained CNN, evaluation metrics, confusion matrix analysis, and single-image inference.

> 🎓 **Learning context:** This project was built with a personal educational goal, learning the fundamentals of Machine Learning and PyTorch, including tensors, convolutional architectures, the training/evaluation loop, backpropagation and model evaluation. It was developed with the support of a generative AI assistant.

## Table of Contents

- [About the Dataset](#about-the-dataset)
- [Pipeline Overview](#pipeline-overview)
- [Environment Setup](#environment-setup)
- [Running the Analysis](#running-the-analysis)
- [Repository Structure](#repository-structure)
- [Viewing Results](#viewing-results)
- [Results](#results)
- [What I Learned](#what-i-learned)
- [License](#license)

## About the Dataset

The project uses the **MNIST** dataset, one of the most widely used introductory benchmarks in computer vision, consisting of:

- 60,000 training images and 10,000 test images
- Grayscale images, 28×28 pixels
- 10 classes, corresponding to handwritten digits 0 through 9

The dataset is downloaded automatically via `torchvision.datasets.MNIST` and is not versioned in this repository.

Official dataset reference: [MNIST database (Yann LeCun et al.)](http://yann.lecun.com/exdb/mnist/)

## Pipeline Overview

```
Raw MNIST dataset (downloaded via torchvision)
│
▼
Preprocessing (ToTensor + Normalize)
│
▼
Dataset exploration & sample visualization
│
▼
CNN architecture (Conv2d → ReLU → MaxPool2d, twice → Flatten → Linear)
│
▼
Training loop (CrossEntropyLoss + Adam optimizer)
│
▼
Training curves (loss & accuracy per epoch)
│
▼
Evaluation on test set (model.eval() + torch.no_grad())
│
▼
Confusion matrix + correct/incorrect prediction analysis
│
▼
Model saving (state_dict) + single-image inference
```

Each stage of this pipeline corresponds to one module in the [`src/`](./src) folder (see [Repository Structure](#repository-structure)).

## Environment Setup

This project was run using the following stack:

```
Python → venv → pip → PyTorch / torchvision / NumPy / Matplotlib / scikit-learn
```

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Verify the install:

```bash
python -c "import torch; print(torch.__version__)"
```

## Running the Analysis

Clone this repository, then move into it:

```bash
git clone https://github.com/pedrocambui06/handwritten-digit-classification.git
cd handwritten-digit-classification
```

Activate your virtual environment (see above), then run the modules **in order**. Each one is commented and mirrors one stage of the pipeline described above.

| Step | Module | What it does |
|------|--------|---------------|
| 1 | `src/data.py` | Downloads MNIST, applies preprocessing, and builds DataLoaders |
| 2 | `src/utils.py` | Inspects dataset statistics and visualizes sample images |
| 3 | `src/model.py` | Defines the CNN architecture |
| 4 | `src/train.py` | Trains the model, plots training curves, and saves the trained weights |
| 5 | `src/evaluate.py` | Evaluates the model on the test set and generates the confusion matrix and prediction visualizations |
| 6 | `src/infer.py` | Loads the saved model and runs inference on a single test image |

You can run the full pipeline with:

```bash
python -m src.train
python -m src.evaluate
python -m src.infer
```

> ⚠️ `src/evaluate.py` currently retrains the model before evaluating it, since model persistence was only introduced in a later stage of the project.

## Repository Structure

```
handwritten-digit-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                # MNIST dataset (not versioned)
├── src/
│ ├── config.py          # Project-wide constants
│ ├── data.py            # Dataset loading & DataLoaders
│ ├── model.py           # CNN architecture
│ ├── train.py           # Training loop
│ ├── evaluate.py        # Evaluation & prediction analysis
│ ├── infer.py           # Model saving, loading & inference
│ └── utils.py           # Visualization utilities
├── models/
│ └── cnn_model.pth      # Saved trained model weights
└── results/
└── figures/             # Generated plots and visualizations
```

## Viewing Results

All generated figures are plain PNG images and can be opened with any image viewer. They are located in `results/figures/`:

| File | Content |
|------|----------|
| `sample_images.png` | Grid of example MNIST digits with their true labels |
| `training_loss.png` | Training loss curve across epochs |
| `training_accuracy.png` | Training accuracy curve across epochs |
| `confusion_matrix.png` | Actual vs. predicted class heatmap on the test set |
| `predictions.png` | Sample test images with actual vs. predicted labels |
| `incorrect_predictions.png` | Examples of misclassified test images |

## Results

| Hyperparameter / Metric | Value |
|---------------------------|-------|
| Epochs                      | 5   |
| Batch size                    | 64 |
| Learning rate                   | 0.001 |
| Optimizer                         | Adam |
| Device                              | CPU |
| Test Accuracy                        | 99.02% |
| Test Loss                              | 0.0295 |

<img src="images/test_loss_and_accuracy.png" width="500" alt="Test Loss and Accuracy">

### Confusion matrix analysis:
Overall, the model performs very well across all classes, with most digits achieving over 950 correct predictions out of roughly 1,000 test samples per class.
Digit 1 is the most accurately classified (1,128 correct, only 7 total misclassifications), followed closely by digit 0 (974 correct).
The most frequent confusion by far is actual 3 predicted as 5 (27 cases) — a systematic error, likely driven by the visual similarity between a rounded, open-top 3 and a 5 written with a curved lower loop.
A secondary, smaller pattern is actual 4 predicted as 9 (9 cases), consistent with the well-known visual overlap between an open-top 4 and a 9.
Digit 6 shows the most scattered error pattern, with small misclassification counts spread across several different classes (0, 1, 2, 4, 5, 8) rather than concentrated on a single confusable digit, suggesting these errors are closer to random noise than a systematic visual confusion.

<img src="results/figures/confusion_matrix.png" width="500" alt="Confusion Matrix">

### Error analysis:
Examining individual misclassified examples confirms the patterns observed in the confusion matrix.
The first sample (top-left), labeled as a 3 but predicted as 5, shows a digit written with a rounded, almost closed loop at the bottom — visually very close to how a 5 is typically drawn, making the error understandable rather than a model failure.
Similarly, two other samples labeled 3 and predicted as 5 show the same rounded-bottom writing style, reinforcing that this specific confusion is driven by handwriting style rather than random noise.
Another clear case is a digit labeled 4 but predicted as 9: the 4 is written with a fully closed top loop, which visually resembles a 9 far more than a typical open-top 4.
These examples suggest that most of the model's errors are not arbitrary — they occur on genuinely ambiguous handwriting samples that could plausibly be misread by a human as well.

<img src="results/figures/incorrect_predictions.png" width="500" alt="Incorrect Predictions">

## What I Learned

This project was my hands-on introduction to several concepts I can now build on for future work:

- What tensors are and how images are represented as them in PyTorch
- Building and training a Convolutional Neural Network from scratch with `nn.Module`
- The full training loop: forward pass, loss calculation, backpropagation, optimizer step
- The distinction between training mode and evaluation mode (`model.train()` vs `model.eval()` + `torch.no_grad()`)
- Interpreting a confusion matrix and identifying patterns in model errors
- Saving and loading model weights (`state_dict`) for reuse without retraining

---

Created by [Pedro C. Martins](https://github.com/pedrocambui06)

Feel free to open issues, propose improvements, or use this project for educational purposes
