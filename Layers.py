# -*- coding: utf-8 -*-
import numpy as np

# Importation de toutes les opérations fonctionnelles du moteur mathématique core_ops
from Core_ops import (
    conv_forward, conv_backward,
    pooling_for, pooling_back,
    GAP_for, GAP_back,
    dense_forward, dense_backward,
    leaky_relu_for, leaky_relu_back,
    gelu_for, gelu_back,
    sigmoid_for, sigmoid_back,
    softmax_for, softmax_back
)

# ==============================================================================
# Base Layer Blueprint
# ==============================================================================

class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, delta_apres):
        raise NotImplementedError


# ==============================================================================
# 1. Convolutional Layer
# ==============================================================================

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.stride = stride
        self.m = kernel_size
        
        # Initialisation He / Kaiming (Convention: C_out, m, m, C_in)
        scale = np.sqrt(2.0 / (kernel_size * kernel_size * in_channels))
        self.K = np.random.randn(out_channels, kernel_size, kernel_size, in_channels) * scale
        self.B = np.zeros(out_channels)
        
        self.dK = None
        self.dB = None
        self.I_cache = None

    def forward(self, I):
        self.I_cache = I
        return conv_forward(I, self.K, self.B, stride=self.stride)

    def backward(self, delta_apres):
        delta_in, self.dK, self.dB = conv_backward(delta_apres, self.I_cache, self.K, stride=self.stride)
        return delta_in


# ==============================================================================
# 2. Max Pooling Layer
# ==============================================================================

class MaxPool2D(Layer):
    def __init__(self, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
        self.mask_cache = None
        self.shape_in_cache = None

    def forward(self, I):
        self.shape_in_cache = I.shape
        out, self.mask_cache = pooling_for(I, self.pool_size)
        return out

    def backward(self, delta_apres):
        return pooling_back(delta_apres, self.mask_cache, self.shape_in_cache, self.pool_size)


# ==============================================================================
# 3. Global Average Pooling (GAP) Layer
# ==============================================================================

class GlobalAvgPool(Layer):
    def __init__(self):
        super().__init__()
        self.shape_in_cache = None

    def forward(self, I):
        self.shape_in_cache = I.shape
        return GAP_for(I)

    def backward(self, delta_apres):
        return GAP_back(delta_apres, self.shape_in_cache)


# ==============================================================================
# 4. Dense / Fully-Connected Layer
# ==============================================================================

class Dense(Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Initialisation He
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.B = np.zeros(out_features)
        
        self.dW = None
        self.dB = None
        self.X_cache = None

    def forward(self, X):
        self.X_cache = X
        return dense_forward(X, self.W, self.B)

    def backward(self, delta_apres):
        delta_in, self.dW, self.dB = dense_backward(delta_apres, self.X_cache, self.W)
        return delta_in


# ==============================================================================
# 5. Activation Layer
# ==============================================================================

class Activation(Layer):
    def __init__(self, mode='leaky_relu', alpha=0.01):
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.Z_cache = None
        self.A_cache = None

    def forward(self, Z):
        self.Z_cache = Z
        if self.mode == 'leaky_relu': 
            return leaky_relu_for(Z, self.alpha)
        elif self.mode == 'gelu':      
            return gelu_for(Z)
        elif self.mode == 'sigmoid':
            self.A_cache = sigmoid_for(Z)
            return self.A_cache
        elif self.mode == 'softmax':
            self.A_cache = softmax_for(Z)
            return self.A_cache

    def backward(self, delta_apres):
        if self.mode == 'leaky_relu': 
            return leaky_relu_back(delta_apres, self.Z_cache, self.alpha)
        elif self.mode == 'gelu':      
            return gelu_back(delta_apres, self.Z_cache)
        elif self.mode == 'sigmoid':
            # sigmoid_back attend A = sigmoid(Z) (la sortie), pas Z (l'entrée)
            return sigmoid_back(delta_apres, self.A_cache)
        elif self.mode == 'softmax':
            # softmax_back attend S = softmax(Z) (la sortie), pas Z (l'entrée)
            return softmax_back(delta_apres, self.A_cache)
