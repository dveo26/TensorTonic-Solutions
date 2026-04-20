import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    if not np.allclose(sum(p),1):
        raise ValueError("Probabilities must sum to 1")
    ans=0.0
    for i in range(len(x)):
        ans+=(x[i]*p[i])
    return ans
