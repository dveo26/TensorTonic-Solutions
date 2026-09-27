import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    """
    Returns a dictionary with pmf, mean, and variance.
    """
    x_arr = np.array(x)
    
    # Calculate the entire PMF array at once without a loop
    pmf = (p ** x_arr) * ((1 - p) ** (1 - x_arr))
    
    return {
        'pmf': pmf,
        'mean': float(p),
        'variance': float(p * (1 - p))
    }