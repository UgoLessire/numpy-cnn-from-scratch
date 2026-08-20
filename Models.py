# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 19:33:57 2026

@author: ugole
"""

import numpy as np
import matplotlib.pyplot as plt
from Layers import Conv2D, MaxPool2D, GlobalAvgPool, Dense, Activation

class CNN:
    def __init__(self, architecture):
        """
        Parses an architecture configuration structure and builds the network.
        
        Parameters:
        -----------
        architecture : list of tuples
            e.g. [
                ("conv", 3, 16, 3, 1, True),   # 3->16 channels, kernel 3, stride 1 + MaxPool
                ("activation", "leaky_relu"),
                ("conv", 16, 32, 3, 2, False), # 16->32 channels, kernel 3, stride 2 (Strided Conv)
                ("activation", "leaky_relu"),
                ("gap",),                      # Global Average Pooling -> outputs 32 features
                ("dense", 32, 10),             # Dense classification head -> 10 classes
                ("activation", "softmax")
            ]
        """
        self.layers = []
        
        for layer_cfg in architecture:
            layer_type = layer_cfg[0].lower()
            
            if layer_type == "conv":
                _, in_c, out_c, k_size, stride, use_maxpool = layer_cfg
                self.layers.append(Conv2D(in_c, out_c, kernel_size=k_size, stride=stride))
                if use_maxpool:
                    self.layers.append(MaxPool2D(pool_size=2))
                    
            elif layer_type == "dense":
                _, in_f, out_f = layer_cfg
                self.layers.append(Dense(in_f, out_f))
                
            elif layer_type == "gap":
                self.layers.append(GlobalAvgPool())
                
            elif layer_type == "activation":
                _, mode = layer_cfg
                self.layers.append(Activation(mode=mode))

    def forward(self, X):
        """Sequential Forward Pass."""
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, delta_loss):
        """Sequential Backward Pass in reverse order."""
        delta = delta_loss
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
        return delta


# ==============================================================================
# Simple Optimizer (Stochastic Gradient Descent)
# ==============================================================================

class SGDOptimizer:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def step(self):
        """Updates trainable parameters (K, B, W) across all layers."""
        for layer in self.model.layers:
            # Update Conv2D weights and biases
            if isinstance(layer, Conv2D):
                layer.K -= self.lr * layer.dK
                layer.B -= self.lr * layer.dB
            # Update Dense weights and biases
            elif isinstance(layer, Dense):
                layer.W -= self.lr * layer.dW
                layer.B -= self.lr * layer.dB
                
class CategoricalCrossEntropy:
    """Pour la classification multi-classes (ex: 10 classes avec Softmax)"""
    def __init__(self):
        self.probs = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        """
        y_pred : probabilités sorties par le Softmax (Batch, N_classes)
        y_true : étiquettes en one-hot encoding (Batch, N_classes)
        """
        # Clip pour éviter log(0) qui renvoie NaN
        self.probs = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.y_true = y_true
        
        # Calcul de la perte moyenne sur le batch
        loss = -np.sum(self.y_true * np.log(self.probs)) / y_pred.shape[0]
        return loss

    def backward(self):
        """
        Calcule dL/dA (gradient par rapport à la sortie du Softmax),
        PAS le raccourci combiné dL/dZ = A - Y.
        C'est la couche Activation(mode='softmax') qui applique ensuite
        le vrai jacobien softmax_back pour obtenir dL/dZ = (A - Y) / N.
        Si on renvoyait directement (A - Y) / N ici, le gradient softmax
        serait appliqué deux fois (bug de double-dérivation).
        """
        N = self.probs.shape[0]
        return -(self.y_true / self.probs) / N


class BinaryCrossEntropy:
    """Pour la classification binaire (1 seule sortie avec Sigmoid)"""
    def __init__(self):
        self.probs = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        """
        y_pred : probabilités sorties par la Sigmoid (Batch, 1)
        y_true : étiquettes binaires 0 ou 1 (Batch, 1)
        """
        self.probs = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.y_true = y_true
        
        loss = -np.mean(
            self.y_true * np.log(self.probs) + (1 - self.y_true) * np.log(1 - self.probs)
        )
        return loss

    def backward(self):
        """
        Calcule dL/dA (gradient par rapport à la sortie de la Sigmoid),
        PAS le raccourci combiné dL/dZ = A - Y (même raison que pour
        CategoricalCrossEntropy : sigmoid_back applique ensuite le vrai
        jacobien pour obtenir dL/dZ = (A - Y) / N).
        """
        N = self.probs.shape[0]
        return (-(self.y_true / self.probs) + (1 - self.y_true) / (1 - self.probs)) / N


class MeanSquaredError:
    """Loss for bounding box coordinate regression."""
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return 0.5 * np.mean((y_pred - y_true) ** 2)

    def backward(self):
        N = self.y_pred.shape[0]
        return (self.y_pred - self.y_true) / N
    
    
class Trainer:
    """
    Handles training, evaluation, and history visualization for CNN models.
    """
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [],  'val_acc': []
        }

    # In models.py -> Inside the Trainer class:
    
    def save_weights(self, filename="model_weights.npz"):
        """Saves all layer parameters (K, B, W) into an NPZ archive."""
        weights_dict = {}
        for idx, layer in enumerate(self.model.layers):
            layer_name = f"layer_{idx}_{layer.__class__.__name__}"
            if hasattr(layer, 'K'):
                weights_dict[f"{layer_name}_K"] = layer.K
            if hasattr(layer, 'W'):
                weights_dict[f"{layer_name}_W"] = layer.W
            if hasattr(layer, 'B'):
                weights_dict[f"{layer_name}_B"] = layer.B
                
        np.savez(filename, **weights_dict)
        print(f"\n[+] Model weights, kernels, and biases successfully saved to '{filename}'.")

    def fit(self, X_train, y_train, epochs=10, batch_size=32, X_val=None, y_val=None):
        num_samples = X_train.shape[0]
        
        for epoch in range(1, epochs + 1):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            running_loss = 0.0
            correct_predictions = 0

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                y_pred = self.model.forward(X_batch)
                loss = self.criterion.forward(y_pred, y_batch)
                running_loss += loss * (end_idx - start_idx)

                if y_batch.shape[1] == 4:
                    correct_predictions += np.sum(np.sqrt(np.sum((y_pred - y_batch)**2, axis=1)))
                else:
                    preds = np.argmax(y_pred, axis=1)
                    targets = np.argmax(y_batch, axis=1)
                    correct_predictions += np.sum(preds == targets)

                delta_loss = self.criterion.backward()
                self.model.backward(delta_loss)
                self.optimizer.step()

            epoch_loss = running_loss / num_samples
            epoch_metric = correct_predictions / num_samples

            self.history['train_loss'].append(epoch_loss)
            self.history['train_acc'].append(epoch_metric)

            val_info = ""
            if X_val is not None and y_val is not None:
                val_loss, val_metric = self.evaluate(X_val, y_val)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_metric)
                
                metric_label = "coord_err" if y_val.shape[1] == 4 else "acc"
                val_info = f" - val_loss: {val_loss:.4f} - val_{metric_label}: {val_metric:.4f}"

            metric_label = "coord_err" if y_train.shape[1] == 4 else "acc"
            print(f"Epoch [{epoch}/{epochs}] - loss: {epoch_loss:.4f} - {metric_label}: {epoch_metric:.4f}{val_info}")

        # ---------------------------------------------------------------------
        # Final Accuracy Evaluation & Interactive Prompt
        # ---------------------------------------------------------------------
        final_acc = self.history['val_acc'][-1] if self.history['val_acc'] else self.history['train_acc'][-1]
        print("\n==================================================")
        print(f"  TRAINING COMPLETED | Final Accuracy: {final_acc * 100:.2f}%")
        print("==================================================")
        
        user_response = input("Sufficient accuracy ? (YES/NO): ").strip()
        if user_response.upper() == "YES":
            self.save_weights("model_weights.npz")
        else:
            print("\n[-] Weights were not saved.")
    
    def evaluate(self, X, y):
        """Evaluates model performance without updating weights."""
        y_pred = self.model.forward(X)
        loss = self.criterion.forward(y, y_pred)
        
        if y.shape[1] == 4:  # Bounding box regression
            metric = np.mean(np.sqrt(np.sum((y_pred - y)**2, axis=1)))
        else:  # Classification
            preds = np.argmax(y_pred, axis=1)
            targets = np.argmax(y, axis=1)
            metric = np.mean(preds == targets)
            
        return loss, metric       
    
    def plot_metrics(self):
        """Plots Training vs Validation Loss and Accuracy side-by-side."""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Loss curves (Overfitting Check)
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], 'b-o', label='Train Loss')
        if self.history['val_loss']:
            plt.plot(epochs, self.history['val_loss'], 'r-o', label='Test/Val Loss')
        plt.title('Loss Curve (Overfitting Evaluation)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_acc'], 'b-o', label='Train Accuracy')
        if self.history['val_acc']:
            plt.plot(epochs, self.history['val_acc'], 'r-o', label='Test/Val Accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()