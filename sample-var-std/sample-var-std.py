import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write 
    n=len(x)
    mean=np.mean(x)
    sum_x=0
    for i in range(n):
        sum_x+=((x[i]-mean)**2)
    variance=sum_x/(n-1)
    deviation=np.sqrt(variance)

    return {'variance':float(variance),'standard_deviation':float(deviation)}