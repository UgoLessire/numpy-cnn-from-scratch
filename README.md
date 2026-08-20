# Modular Deep Learning Framework from Scratch (Pure NumPy)

A framework-free, modular Deep Learning library built entirely from scratch in Python using pure NumPy.

This framework allows you to build, train, and evaluate convolutional networks using vectorized operations, automated backpropagation, and interactive weight serialization.

---

## Project Directory Structure

- core_ops.py: Functional engine handling vectorized operations and im2col patch extraction.
- layers.py: Object-oriented layer implementations (Conv2D, Dense, GlobalAvgPool, Activation).
- models.py: Sequential CNN model wrapper, SGD optimizer, loss functions (CCE, BCE, MSE), and training engine.
- test_digits.py: Main demonstration script for handwritten digit classification (0-9).
- README.md: Project documentation.

---

## Key Features

- Modular Architecture: Easily stack layers (Convolution, Activation, Pooling, Dense) in a simple list definition.
- Framework-Free Implementation: Pure NumPy operations without relying on PyTorch or TensorFlow.
- Unified Loss Engine: Supports multi-class classification (Categorical Cross-Entropy), binary classification (Binary Cross-Entropy), and regression tasks (Mean Squared Error).
- Automated Training Pipeline: Includes data batching, validation monitoring, metric logging, and loss visualization.
- Interactive Model Serialization: Automatically prompts the user at the end of training to export optimized parameters to a compressed file.

---

## Quickstart and User Guide

### 1. Prerequisites and Installation

Ensure Python is installed along with the required dependencies:

pip install numpy matplotlib scikit-learn

### 2. Running the Digit Classification Demo

Execute the training script to benchmark the framework on the 8x8 handwritten digits dataset:

python test_digits.py

### 3. Interactive Weight Serialization

When training completes, the terminal displays the final performance metrics and asks if you want to save the trained model:

==================================================
  TRAINING COMPLETED | Final Accuracy: 87.22%
==================================================
Sufficient accuracy ? (YES/NO): YES

[+] Model weights, kernels, and biases successfully saved to 'model_weights.npz'.

- Type YES to save all kernels, weights, and biases into a model_weights.npz archive.
- Type NO to exit without saving.

---

## Minimal Code Example

Below is a basic example of how to build, compile, and train a network using this library:

import numpy as np
from models import CNN, SGDOptimizer, CategoricalCrossEntropy, Trainer

# 1. Define Network Architecture
architecture = [
    ("conv", 1, 8, 3, 1, False),
    ("activation", "leaky_relu"),
    ("gap",),
    ("dense", 8, 10),
    ("activation", "softmax")
]

# 2. Instantiate Components
model = CNN(architecture)
criterion = CategoricalCrossEntropy()
optimizer = SGDOptimizer(model, lr=0.05)
trainer = Trainer(model, criterion, optimizer)

# 3. Train Model
trainer.fit(X_train, y_train, epochs=15, batch_size=32, X_val=X_test, y_val=y_test)
trainer.plot_metrics()

---

## References

- Goodfellow, I., Bengio, Y., and Courville, A. (2016). Deep Learning. MIT Press.
- Dumoulin, V., and Visin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv:1603.07285.
