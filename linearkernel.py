import numpy as np

def linear_kernel(x1, x2):
    """
    Linear kernel.
    
    Parameters:
        x1: array-like, has dimension dxn
        x2: array-like, has dimension dxn

    Returns:
        K: the kernel matrix; array, has dimension nxn
    """
    # Linear kernel is simply the dot product between the inputs
    lin_kernel = np.dot(x1.T, x2)
    
    return lin_kernel