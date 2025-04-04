import numpy as np

def negative_log_predictive_density(y_tests, y_preds, variances):
    """
    Compute the negative log predictive density (NLPD) using the standardized data.
    
    The formula is:
    NLPD = (1/n) sum_i (1/2 * log(2π * σ_i^2) + (y_i - μ_i)^2 / (2 * σ_i^2))
    
    Parameters:
        y_tests: True target values
        y_preds: Predicted mean values
        variances: Predicted variances for each test point
        
    Returns:
        nlpd: The negative log predictive density
    """
    n = len(y_tests)
    
    # Ensure the arrays are the right shape
    y_tests = np.array(y_tests).flatten()
    y_preds = np.array(y_preds).flatten()
    variances = np.array(variances).flatten()
    
    # Compute the NLPD
    log_term = 0.5 * np.log(2 * np.pi * variances)
    squared_error_term = ((y_tests - y_preds) ** 2) / (2 * variances)
    
    # Sum the terms and divide by n
    nlpd = np.mean(log_term + squared_error_term)
    
    return nlpd