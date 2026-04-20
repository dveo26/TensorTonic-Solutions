import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x)

    if rng is None:
        rng = np.random.default_rng()

    # create binary mask (0 or 1)
    keep = (rng.random(x.shape) < 1-p)

    # create pattern: 0 OR 1/(1-p)
    pattern = keep.astype(float) / (1 - p)

    # apply dropout
    out = x * pattern

    return out, pattern