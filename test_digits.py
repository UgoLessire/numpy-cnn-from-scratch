# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 19:42:54 2026

@author: ugole
"""
# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Ensure local imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Models import CNN, SGDOptimizer, CategoricalCrossEntropy, Trainer

def train_on_digits():
    print("=== TRAINING CNN ON HANDWRITTEN DIGITS DATASET (0-9) ===\n")

    # 1. Load and prepare dataset (1797 samples of 8x8 images)
    digits = load_digits()
    X_raw = digits.images  # Shape: (1797, 8, 8)
    y_raw = digits.target  # Shape: (1797,)

    # Normalize pixel values to range [0, 1] and add Channel dimension -> (N, 8, 8, 1)
    X = (X_raw / 16.0)[:, :, :, np.newaxis]
    
    # One-hot encode targets for 10 classes
    y_onehot = np.eye(10)[y_raw]

    # Split into 80% Train / 20% Test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42
    )

    print(f"Dataset Loaded:")
    print(f" -> Train shape: {X_train.shape}")
    print(f" -> Test shape : {X_test.shape}\n")

    # 2. Define Architecture for 8x8 Single-Channel Images
    # Note: We skip MaxPool because 8x8 is already small
    architecture = [
        ("conv", 1, 16, 3, 1, False),     # Output: (8, 8, 8)
        ("activation", "leaky_relu"),
        ("conv", 16, 32, 3, 1, False),    # Output: (8, 8, 16)
        ("activation", "leaky_relu"),
        ("conv", 32, 64, 3, 1, False),    # Output: (8, 8, 16)
        ("activation", "leaky_relu"),
        ("gap",),                        # Output: (16,)
        ("dense", 64, 10),               # Output: (10,)
        ("activation", "softmax")
    ]

    # 3. Instantiate Model, Loss, Optimizer & Trainer
    model = CNN(architecture)
    criterion = CategoricalCrossEntropy()
    optimizer = SGDOptimizer(model, lr=0.01)
    trainer = Trainer(model, criterion, optimizer)

    # 4. Train model over 15 Epochs
    print("[1/2] Starting training loop...\n")
    trainer.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        X_val=X_test,
        y_val=y_test
    )

    # 5. Plot Loss & Accuracy Curves
    print("\n[2/2] Displaying training vs. test metrics...")
    trainer.plot_metrics()

if __name__ == "__main__":
    train_on_digits()