import numpy as np

def masked_language_model_logits(hidden_states, mask_positions, W, b):
    masked_states = hidden_states[mask_positions]
    return masked_states @ W + b

def predict_from_logits(logits):
    return np.argmax(logits, axis=1)
