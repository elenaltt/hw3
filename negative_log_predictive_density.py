import numpy as np
'''
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
    # Convert all inputs to flat numpy arrays
    y_tests = np.array(y_tests, dtype=np.float64).flatten()
    y_preds = np.array(y_preds, dtype=np.float64).flatten()
    variances = np.array(variances, dtype=np.float64).flatten()
    
    # Make sure we have matching sizes
    assert len(y_tests) == len(y_preds) == len(variances), "Input arrays must have the same length"
    
    # Ensure numerical stability by setting a minimum variance value
    variances = np.maximum(variances, 1e-10)
    
    # Number of test points
    n = len(y_tests)
    
    # Calculate NLPD for each test point
    nlpd_values = np.zeros(n)
    for i in range(n):
        # Log term: 0.5 * log(2π * σ²)
        log_term = 0.5 * np.log(2 * np.pi * variances[i])
        
        # Squared error term: (y - μ)² / (2 * σ²)
        squared_error = ((y_tests[i] - y_preds[i])**2) / (2 * variances[i])
        
        # Sum for this test point
        nlpd_values[i] = log_term + squared_error
    
    # Average over all test points
    nlpd = np.mean(nlpd_values)
    
    return nlpd
'''


def negative_log_predictive_density(y_tests, y_preds, variances):
    """
    Compute the negative log predictive density (NLPD) for Gaussian Processes.

    Formula:
        NLPD = (1/n) ∑ [ 0.5 * log(2πσ²_i) + (y_i - μ_i)^2 / (2σ²_i) ]

    Note:
        The inputs are already standardized in the hidden test set.
        DO NOT standardize again inside this function.

    Parameters:
        y_tests (array-like): True (standardized) values
        y_preds (array-like): Predicted mean values
        variances (array-like): Predicted variances

    Returns:
        float: Mean negative log predictive density
    """
    y_tests = np.array(y_tests, dtype=np.float64).flatten()
    y_preds = np.array(y_preds, dtype=np.float64).flatten()
    variances = np.array(variances, dtype=np.float64).flatten()

    # Check shape
    assert len(y_tests) == len(y_preds) == len(variances)

    # Numerical stability
    variances = np.maximum(variances, 1e-10)

    # Compute NLPD
    term1 = 0.5 * np.log(2 * np.pi * variances)
    term2 = ((y_tests - y_preds) ** 2) / (2 * variances)

    return np.mean(term1 + term2)
