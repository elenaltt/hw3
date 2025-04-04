import numpy as np
import math
from gaussianprocess import GaussianProcess
from sklearn.model_selection import KFold
from standardize import standardize_targets, unstandardize_targets

def crossvalidate(xTr, yTr, ktype, noise_vars, paras):
    """
    INPUT:	
      xTr : dxn input matrix
      yTr : 1xn input continuous target
      ktype : kernel type: 'linear', 'rbf', or 'polynomial'
      noise_vars : list of noise variance values to try
      paras: list of kernel parameters to try
      
    Output:
      best_noise: best performing noise variance
      bestP: best performing kernel parameter
      lowest_error: best performing validation error (mean squared error)
      errors: a matrix where errors[i,j] is the validation error with parameters paras[i] and noise_vars[j]
      
    Trains a GP regressor for all combinations of noise_vars and paras and identifies the best setting.
    """

    # YOUR CODE HERE

    
    return best_noise, bestP, lowest_error, errors
