import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    if not x:
        return {"mean": None, "median": None, "mode": None}
        
    x_arr = np.array(x)
    
   
    mean_val = np.mean(x_arr)
 
    median_val = np.median(x_arr)
    
    values, counts = np.unique(x_arr, return_counts=True)
    max_index = np.argmax(counts)
    mode_val = values[max_index]
    
    return {
        "mean": float(mean_val),
        "median": float(median_val),
        "mode": float(mode_val)  
    }

