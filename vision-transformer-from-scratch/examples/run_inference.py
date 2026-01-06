import os
import sys
import numpy as np
# Ensure project root is on the Python path so `from vit.vit import ...` works when running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vit.vit import vit_forward_pipeline

# Image
image = np.random.randn(16, 16, 3).astype(np.float32)
patch_size = 4

# Derived values
num_patches = (16 // patch_size) ** 2
patch_dim = patch_size * patch_size * 3
num_classes = 10

# Positional embedding (learnable in real models)
position_embeddings = np.random.randn(num_patches, patch_dim).astype(np.float32)

# Class token
class_token = np.random.randn(1, patch_dim).astype(np.float32)

# Encoder parameters (1 layer demo)
encoder_params = [{
    'attn_params': {
        'W_q': np.random.randn(patch_dim, patch_dim).astype(np.float32),
        'W_k': np.random.randn(patch_dim, patch_dim).astype(np.float32),
        'W_v': np.random.randn(patch_dim, patch_dim).astype(np.float32),
        'W_o': np.random.randn(patch_dim, patch_dim).astype(np.float32),
        'num_heads': 4,
        'mask': None
    },
    'mlp_params': {
        'W1': np.random.randn(patch_dim, patch_dim * 4).astype(np.float32),
        'b1': np.zeros(patch_dim * 4, dtype=np.float32),
        'W2': np.random.randn(patch_dim * 4, patch_dim).astype(np.float32),
        'b2': np.zeros(patch_dim, dtype=np.float32)
    },
    'ln1_params': {
        'gamma': np.ones(patch_dim),
        'beta': np.zeros(patch_dim)
    },
    'ln2_params': {
        'gamma': np.ones(patch_dim),
        'beta': np.zeros(patch_dim)
    }
}]

# Classification head
W_cls = np.random.randn(patch_dim, num_classes).astype(np.float32)
b_cls = np.random.randn(num_classes).astype(np.float32)

# Forward pass
logits = vit_forward_pipeline(
    image,
    patch_size,
    position_embeddings,
    class_token,
    encoder_params,
    W_cls,
    b_cls
)

print("Output logits:", logits)
