import numpy as np
from l2distance import l2_distance

# Define the kernel function (RBF kernel)
def rbf_kernel(x1, x2, kpar):
    """
    Radial Basis Function (RBF) kernel (Gaussian kernel).
    
    Parameters:
        x1: array-like, has dimensions (d,n)
        x2: array-like, has dimensions (d,n)
        kpar: inverse kernel width

    Returns:
        K: the kernel matrix; array-like, has dimensions (n,n)
    """
    # YOUR CODE HERE

    return rbf_kern