import numpy as np
from .patch_embedding import patch_embedding
from .positional_embedding import add_positional_embedding
from .encoder import vit_encoder_forward
from .classification_head import vit_classification_head

def vit_forward_pipeline(
    image,
    patch_size,
    position_embeddings,
    class_token,
    encoder_params,
    W_cls,
    b_cls
):
    # 1. Patch embedding
    patches = patch_embedding(image, patch_size)

    # 2. Add positional embeddings
    patches = add_positional_embedding(patches, position_embeddings)

    # 3. Prepend class token
    x = np.vstack([class_token, patches])

    # 4. Transformer encoder
    x = vit_encoder_forward(x, encoder_params)

    # 5. Classification head
    logits = vit_classification_head(x, W_cls, b_cls)
    return logits
