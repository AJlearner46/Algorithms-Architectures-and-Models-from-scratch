import numpy as np

def build_input_embeddings(token_ids, segment_ids,
                           position_embeddings,
                           token_embeddings,
                           segment_embeddings):
    token_emb = token_embeddings[token_ids]
    segment_emb = segment_embeddings[segment_ids]

    return (token_emb + segment_emb + position_embeddings).astype(np.float32)
