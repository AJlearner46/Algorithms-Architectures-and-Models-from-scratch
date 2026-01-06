import numpy as np

from .attention import multi_head_attention_forward
from .mlp import mlp_forward
from .layer_norm import layer_norm

def vit_encoder_layer(x, params):
    # LayerNorm → Attention → Residual
    x_norm = layer_norm(x, **params['ln1_params'])
    attn_out = multi_head_attention_forward(
        x_norm,
        **params['attn_params']
    )
    x = x + attn_out

    # LayerNorm → MLP → Residual
    x_norm = layer_norm(x, **params['ln2_params'])
    mlp_out = mlp_forward(x_norm, **params['mlp_params'])
    x = x + mlp_out

    return x

def vit_encoder_forward(x, layers_params):
    for layer_params in layers_params:
        x = vit_encoder_layer(x, layer_params)
    return x
