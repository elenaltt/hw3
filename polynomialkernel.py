import numpy as np

def polynomial_kernel(x1, x2, kpar):
    """
    Polynomial kernel.
    
    Parameters:
        x1: array-like, has dimensions (d,n)
        x2: array-like, has dimensions (d,n)
        kpar: int, degree of the polynomial

    Returns:
        K: the kernel matrix; array-like, has dimensions (n,n)
    """
    # Polynomial kernel: (x^T * z + 1)^p
    # First compute the dot product
    dot_product = np.dot(x1.T, x2)
    
    # Add 1 and raise to the power p (kpar)
    poly_kernel = (dot_product + 1) ** kpar
    
    return poly_kernel