import numpy as np
from l2distance import l2_distance

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
    # RBF kernel: exp(-gamma * ||x-z||^2)
    # First compute the squared distance matrix
    squared_distances = l2_distance(x1, x2) ** 2
    
    # Apply the RBF formula with gamma = kpar
    rbf_kern = np.exp(-kpar * squared_distances)
    
    return rbf_kern