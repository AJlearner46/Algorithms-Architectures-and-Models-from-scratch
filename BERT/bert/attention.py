import numpy as np

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + (mask * -1e9)

    scores -= np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=-1, keepdims=True)

    return weights @ v
