import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    # Write code here
    dot=float(0)
    for i in range (len(x)):
        dot+=(x[i]*y[i])
        
    return dot