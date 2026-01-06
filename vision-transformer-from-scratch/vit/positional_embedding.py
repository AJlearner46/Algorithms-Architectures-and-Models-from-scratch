import numpy as np

def add_positional_embedding(
    patch_embeddings: np.ndarray,
    position_embeddings: np.ndarray
) -> np.ndarray:
    """
    Add positional embeddings to patch embeddings.

    Args:
        patch_embeddings: (N, D)
        position_embeddings: (N, D)

    Returns:
        (N, D)
    """
    assert patch_embeddings.shape == position_embeddings.shape
    return patch_embeddings + position_embeddings
