import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    attn = softmax(scores)
    return attn @ V

def multi_head_attention_forward(x, W_q, W_k, W_v, W_o, num_heads, mask=None):
    N, D = x.shape
    assert D % num_heads == 0
    d_k = D // num_heads

    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v

    Q = Q.reshape(N, num_heads, d_k)
    K = K.reshape(N, num_heads, d_k)
    V = V.reshape(N, num_heads, d_k)

    heads = []
    for h in range(num_heads):
        head = scaled_dot_product_attention(
            Q[:, h, :],
            K[:, h, :],
            V[:, h, :],
            mask
        )
        heads.append(head)

    concat = np.concatenate(heads, axis=-1)
    return concat @ W_o
