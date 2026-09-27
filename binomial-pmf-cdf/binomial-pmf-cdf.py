import math

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    # Write code here
    combinations=math.comb(n,k)
    pmf=float(combinations*(p**k)*((1-p)**(n-k)))
    cdf=float(0)
    for i in range(k+1):
        cdf+=float((math.comb(n,i))*(p**i)*((1-p)**(n-i)))
    return {
        'pmf':pmf,
        'cdf': cdf
    }