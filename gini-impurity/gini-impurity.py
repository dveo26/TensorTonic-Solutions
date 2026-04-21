import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    n = len(y_left) + len(y_right)
    if n==0:
        return 0.0
    # LEFT
    if len(y_left) == 0:
        gini_l = 0
    else:
        _, counts_l = np.unique(y_left, return_counts=True)
        prob_l = counts_l / len(y_left)
        gini_l = 1 - np.sum(prob_l ** 2)

    # RIGHT
    if len(y_right) == 0:
        gini_r = 0
    else:
        _, counts_r = np.unique(y_right, return_counts=True)
        prob_r = counts_r / len(y_right)
        gini_r = 1 - np.sum(prob_r ** 2)

    # weighted sum
    gini_split = (len(y_left)/n) * gini_l + (len(y_right)/n) * gini_r

    return gini_split