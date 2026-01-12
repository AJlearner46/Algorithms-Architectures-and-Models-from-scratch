import numpy as np
from .attention import scaled_dot_product_attention

def layer_norm(x, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def multi_head_attention(x, w_q, w_k, w_v, w_o, num_heads):
    batch, seq_len, d_model = x.shape
    d_k = d_model // num_heads

    Q = x @ w_q
    K = x @ w_k
    V = x @ w_v

    Q = Q.reshape(batch, seq_len, num_heads, d_k).transpose(0,2,1,3)
    K = K.reshape(batch, seq_len, num_heads, d_k).transpose(0,2,1,3)
    V = V.reshape(batch, seq_len, num_heads, d_k).transpose(0,2,1,3)

    attn = scaled_dot_product_attention(Q, K, V)
    attn = attn.transpose(0,2,1,3).reshape(batch, seq_len, d_model)

    return attn @ w_o

def transformer_encoder_layer(x, attn_weights, ff_weights, num_heads):
    attn_out = multi_head_attention(
        x,
        attn_weights['w_q'],
        attn_weights['w_k'],
        attn_weights['w_v'],
        attn_weights['w_o'],
        num_heads
    )

    x = layer_norm(x + attn_out)

    ff = np.maximum(0, x @ ff_weights['w1'] + ff_weights['b1'])
    ff = ff @ ff_weights['w2'] + ff_weights['b2']

    return layer_norm(x + ff)
