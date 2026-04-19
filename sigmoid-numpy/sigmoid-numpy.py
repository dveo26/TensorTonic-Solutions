import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    ans=0
    denominator=1+1/np.exp(x)
    numerator=1
    ans=numerator/denominator
    return ans