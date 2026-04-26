import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    prob=y_pred[np.arange(len(y_true)),y_true]
    ans=-np.mean(np.log(prob))
    return ans