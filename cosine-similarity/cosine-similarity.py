import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    cosine=float(0)
    dot=float(0)
    norm_a=float(0)
    norm_b=float(0)
    for i in range (len(a)):
        dot+=(a[i]*b[i])
        norm_a+=(a[i]**2)
        norm_b+=(b[i]**2)

    if norm_a==0 or norm_b==0:
        return float(0)
    cosine=dot/(np.sqrt(norm_a)*np.sqrt(norm_b))

    return float(cosine)
    