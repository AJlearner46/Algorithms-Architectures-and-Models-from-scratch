import numpy as np

def gelu(x):
    return 0.5 * x * (1 + np.tanh(
        np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
    ))

def mlp_forward(x, W1, b1, W2, b2):
    x = gelu(x @ W1 + b1)
    return x @ W2 + b2
