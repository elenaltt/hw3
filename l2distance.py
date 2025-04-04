import numpy as np

def l2_distance(X, Z):
    """
    Compute the L2 (Euclidean) distance between two vectors.

    Parameters:
        X: dxn data matrix with n vectors (columns) of dimensionality d
        Z: dxm data matrix with m vectors (columns) of dimensionality d

    Returns:
        Matrix D of size nxm 
        D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
    """
    # Efficient implementation without loops
    # We use the formula: ||x-y||^2 = ||x||^2 + ||y||^2 - 2x^Ty
    
    X_squared_norm = np.sum(X**2, axis=0, keepdims=True).T  # n x 1
    Z_squared_norm = np.sum(Z**2, axis=0, keepdims=True)    # 1 x m
    
    # X is d x n, Z is d x m, so X.T @ Z gives an n x m matrix
    cross_term = np.dot(X.T, Z)  # n x m
    
    # Final formula - ensure non-negative values before sqrt
    squared_dist = X_squared_norm + Z_squared_norm - 2 * cross_term
    
    # Clip small negative values to zero (these are numerical errors)
    squared_dist = np.maximum(squared_dist, 0)
    D = np.sqrt(squared_dist)
    
    return D