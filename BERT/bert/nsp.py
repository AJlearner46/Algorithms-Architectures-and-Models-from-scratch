def next_sentence_prediction_scores(cls_hidden_states, W, b):
    return cls_hidden_states @ W + b
