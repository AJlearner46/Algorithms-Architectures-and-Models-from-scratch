import numpy as np

def mask_selection_logic(token_ids, mask_prob, mask_token_id, vocab_size, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    mask_positions = np.random.rand(len(token_ids)) < mask_prob
    masked = token_ids.copy()

    for i, mask in enumerate(mask_positions):
        if not mask:
            continue

        rand = np.random.rand()
        if rand < 0.8:
            masked[i] = mask_token_id
        elif rand < 0.9:
            masked[i] = np.random.randint(0, vocab_size)
        else:
            pass

    return masked, mask_positions
