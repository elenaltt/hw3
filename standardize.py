import numpy as np

def standardize_targets(y):
    """
    Standardize target values to have zero mean and unit variance.
    
    Parameters:
        y : array-like, shape (1,n)
        
    Returns:
        y_standardized: array of standardized targets
        mean: float, the mean of the original targets
        std: float, the standard deviation of the original targets
    """
    # Calculate mean and standard deviation
    mean = np.mean(y)
    std = np.std(y)
    
    # Standardize the targets
    y_standardized = (y - mean) / std
    
    return y_standardized, mean, std

def unstandardize_targets(y_standardized, mean, std):
    """
    Revert standardization so that the target is on its
    original scale. Makes comparisons between original
    target values and predicted target values more
    interpretable.
    
    Parameters:
        y_standardized : array-like, shape (n_samples,)
        mean: float, the mean of the original targets
        std: float, the standard deviation of the original targets
        
    Returns:
        y : array-like, shape (n_samples,)
    """
    # Reverse the standardization
    y = (y_standardized * std) + mean
    
    return y