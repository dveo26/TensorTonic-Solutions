import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
   """
    y=np.array(y)
    value,counts=np.unique(y,return_counts=True)

    probs=counts/len(y)
    return -np.sum(probs*np.log2(probs))