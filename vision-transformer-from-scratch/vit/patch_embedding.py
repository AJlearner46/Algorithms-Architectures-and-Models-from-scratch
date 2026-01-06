import numpy as np

def patch_embedding(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Extract non-overlapping patches and flatten them.

    Args:
        image: (H, W, C)
        patch_size: int

    Returns:
        patches: (N, patch_dim)
    """
    H, W, C = image.shape
    assert H % patch_size == 0 and W % patch_size == 0

    patches = []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch.flatten())

    return np.array(patches, dtype=np.float32)
