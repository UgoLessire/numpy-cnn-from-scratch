# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:54:39 2026

@author: ugole
"""
import numpy as np 
from numpy.lib.stride_tricks import as_strided

#==============================================================================
# Functions & Convolution Forward
#==============================================================================

# I : 4D Tensor (Batch, H, W, C_in)
# K : 4D Tensor (C_out, m, m, C_in)

def convolution_batch(I, K, DEBUG=False, stride=1):
    batch_size, _, _, c = I.shape
    m = K.shape[1]

    if c != K.shape[3]:
        if DEBUG: print(f"\n[!] ERREUR BROADCASTING : Image({c}) != Filtre({K.shape[3]})")

    p = m // 2
    img = np.pad(I, ((0,0), (p,p), (p,p), (0,0)), mode='constant')

    sB, s0, s1, s2 = img.strides
    h_pad, w_pad = img.shape[1], img.shape[2]

    s_h, s_w = stride if isinstance(stride, tuple) else (stride, stride)
    h_out = (h_pad - m) // s_h + 1
    w_out = (w_pad - m) // s_w + 1

    nouv_shape = (batch_size, h_out, w_out, m, m, c)
    nouv_strides = (sB, s0 * s_h, s1 * s_w, s0, s1, s2)

    fenetre = as_strided(img, shape=nouv_shape, strides=nouv_strides)

    return np.tensordot(fenetre, K, axes=([-3,-2,-1], [1,2,3]))


def conv_forward(I, K, B, DEBUG=False, stride=1):
    """Forward pass for Convolutional Layer (handles both stride=1 and strided convs)."""
    return convolution_batch(I, K, DEBUG, stride=stride) + B


#==============================================================================
# Convolution Backward (Returns delta_in, dK, dB)
#==============================================================================

def conv_backward(delta_apres, I, K, stride=1):
    """
    Unified Convolution Backward Pass.
    Computes delta_in (error transmitted to previous layer) as well as 
    parameter gradients dK and dB.
    
    Parameters:
    -----------
    delta_apres : (Batch, H_out, W_out, C_out)
    I           : (Batch, H_in, W_in, C_in)
    K           : (C_out, m, m, C_in)
    stride      : int or tuple (s_h, s_w)
    
    Returns:
    --------
    delta_in : (Batch, H_in, W_in, C_in) -> Transmitted error
    dK       : (C_out, m, m, C_in)       -> Kernel weight gradients
    dB       : (C_out,)                  -> Bias gradients
    
    """
    batch_size, h_in, w_in, c_in = I.shape
    C_out, m, _, _ = K.shape
    s_h, s_w = stride if isinstance(stride, tuple) else (stride, stride)
    
    # -------------------------------------------------------------------------
    # 1. Compute dK and dB (Parameter Gradients)
    # -------------------------------------------------------------------------
    p = m // 2
    img = np.pad(I, ((0,0), (p,p), (p,p), (0,0)), mode='constant')
    
    sB, s0, s1, s2 = img.strides
    h_out, w_out = delta_apres.shape[1], delta_apres.shape[2]
    
    patch_shape = (batch_size, h_out, w_out, m, m, c_in)
    patch_strides = (sB, s0 * s_h, s1 * s_w, s0, s1, s2)
    patches = as_strided(img, shape=patch_shape, strides=patch_strides)
    
    # dK: Contraction over (Batch, H_out, W_out) -> yields (C_out, m, m, C_in)
    dK = np.tensordot(delta_apres, patches, axes=([0, 1, 2], [0, 1, 2]))
    
    # dB: Sum over (Batch, H_out, W_out) -> yields (C_out,)
    dB = np.sum(delta_apres, axis=(0, 1, 2))
    
    # -------------------------------------------------------------------------
    # 2. Compute delta_in (Scatter-Add Input Error Transmission)
    # -------------------------------------------------------------------------
    delta_pad = np.zeros((batch_size, h_in + 2 * p, w_in + 2 * p, c_in), dtype=delta_apres.dtype)
    
    grad_windows = np.tensordot(delta_apres, K, axes=([3], [0]))
    
    h_idx = (np.arange(h_out)[:, None, None, None] * s_h
             + np.arange(m)[None, None, :, None])
    w_idx = (np.arange(w_out)[None, :, None, None] * s_w
             + np.arange(m)[None, None, None, :])

    h_idx = np.broadcast_to(h_idx, (h_out, w_out, m, m))
    w_idx = np.broadcast_to(w_idx, (h_out, w_out, m, m))

    np.add.at(delta_pad, (slice(None), h_idx, w_idx, slice(None)), grad_windows)
    
    if p > 0:
        delta_in = delta_pad[:, p:-p, p:-p, :]
    else:
        delta_in = delta_pad
        
    return delta_in, dK, dB


#==============================================================================
# Global Average Pooling (GAP)
#==============================================================================

##### FORWARD

def GAP_for(volume):
    return np.mean(volume, axis=(1, 2))


##### BACKWARD

def GAP_back(delta_l, input_shape):
    Batch, H, W, C = input_shape
    delta_reshaped = delta_l[:, None, None, :]
    return np.broadcast_to(delta_reshaped / (H * W), input_shape)

        
#==============================================================================
# Pooling 
#==============================================================================

##### FORWARD

def pooling_for(I, pool_size):
    batch_size, h, w, c = I.shape
    m, n = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
    
    pad_h = (m - h % m) % m
    pad_w = (n - w % n) % n
    if pad_h > 0 or pad_w > 0:
        I_pad = np.pad(I, ((0,0), (0,pad_h), (0,pad_w), (0,0)), mode='constant')
    else:
        I_pad = I

    _, h_pad, w_pad, _ = I_pad.shape
    h_out, w_out = h_pad // m, w_pad // n
    sB, s0, s1, s2 = I_pad.strides

    nouv_shape = (batch_size, h_out, w_out, m, n, c)
    nouv_strides = (sB, s0*m, s1*n, s0, s1, s2)
    
    fenetre = as_strided(I_pad, shape=nouv_shape, strides=nouv_strides)
    max_fenetre = np.max(fenetre, axis=(-3, -2))
    
    masque_6d = (fenetre == max_fenetre[:, :, :, None, None, :])
    masque_pad = masque_6d.transpose(0, 1, 3, 2, 4, 5).reshape(batch_size, h_pad, w_pad, c)
    cache = masque_pad[:, :h, :w, :]
   
    return max_fenetre, cache 


##### BACKWARD

def pooling_back(delta, masque, shape_origine, pool_size):
    m, n = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
    _, H, W, _ = shape_origine
    h_lim, w_lim = (H // m) * m, (W // n) * n
    
    delta_repeat = np.repeat(np.repeat(delta, m, axis=1), n, axis=2)
    delta_final = np.zeros(shape_origine)
    delta_final[:, :h_lim, :w_lim, :] = delta_repeat * masque
    
    return delta_final


# ==============================================================================
# Fonctions d'Activation (Forward & Backward)
# ==============================================================================

##### Sigmoïde
def sigmoid_for(Z):
    Z_clipped = np.clip(Z, -500, 500)
    return 1 / (1 + np.exp(-Z_clipped))

def sigmoid_back(delta_apres, A):
    return delta_apres * A * (1 - A)


##### Softmax
def softmax_for(Z):
    e_Z = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
    return e_Z / np.sum(e_Z, axis=-1, keepdims=True)

def softmax_back(delta_apres, S):
    sum_ds = np.sum(delta_apres * S, axis=-1, keepdims=True)
    return S * (delta_apres - sum_ds)


##### GeLU
def gelu_for(Z):
    return 0.5 * Z * (1 + np.tanh(np.sqrt(2 / np.pi) * (Z + 0.044715 * Z**3)))

def gelu_back(delta_apres, Z):
    cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (Z + 0.044715 * Z**3)))
    pdf = np.exp(-0.5 * Z**2) / np.sqrt(2 * np.pi)
    dZ = cdf + Z * pdf * (1 + 3 * 0.044715 * Z**2)
    return delta_apres * dZ


##### Leaky ReLU
def leaky_relu_for(Z, alpha=0.01):
    return np.where(Z > 0, Z, Z * alpha)

def leaky_relu_back(delta_apres, Z, alpha=0.01):
    dZ = np.ones_like(Z)
    dZ[Z <= 0] = alpha
    return delta_apres * dZ

# ==============================================================================
# Dense / Fully-Connected Layer Operations
# ==============================================================================

def dense_forward(X, W, B):
    """
    
    Parameters:
    -----------
    X : Input matrix format (Batch, in_features)
    W : Weights format (in_features, out_features)
    B : Biases format (out_features,)
    
    Output:
    --------
    Z : Output (Batch, out_features)
    """
    return np.dot(X, W) + B


def dense_backward(delta_apres, X, W):
    """
    
    Parameters:
    -----------
    delta_apres : Gradient coming from the next layer, format (Batch, out_features)
    X : entry, format (Batch, in_features)
    W : weights (in_features, out_features)
    
    Retourne:
    --------
    delta_in : gradient coming from the old layer (Batch, in_features)
    dW : Gradient for the weights W (in_features, out_features)
    dB : Gradient for the biases B (out_features,)
    """
    dW = np.dot(X.T, delta_apres)
    dB = np.sum(delta_apres, axis=0)
    delta_in = np.dot(delta_apres, W.T)
    
    return delta_in, dW, dB



