import numpy as np

def vit_classification_head(encoder_output, W_cls, b_cls):
    cls_token = encoder_output[0]
    return cls_token @ W_cls + b_cls
