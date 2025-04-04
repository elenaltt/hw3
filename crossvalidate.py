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
    """
    # Number of folds for cross-validation
    n_folds = 5
    
    # Initialize an array to store validation errors for all parameter combinations
    if ktype == 'linear':
        # Linear kernel has no hyperparameter, so we only need to consider noise values
        errors = np.zeros((1, len(noise_vars)))
    else:
        errors = np.zeros((len(paras), len(noise_vars)))
    
    # Standardize targets before cross-validation
    y_std, y_mean, y_std_dev = standardize_targets(yTr)
    
    # Initialize KFold cross-validator
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Indices for the n data points
    n = xTr.shape[1]
    indices = np.arange(n)
    
    # For each parameter combination, compute the average validation error over all folds
    if ktype == 'linear':
        # Linear kernel has no hyperparameter, so we only vary noise
        for j, noise in enumerate(noise_vars):
            fold_errors = []
            
            for train_idx, val_idx in kf.split(indices):
                # Extract training and validation data for this fold
                X_train, X_val = xTr[:, train_idx], xTr[:, val_idx]
                y_train, y_val = y_std[:, train_idx], y_std[:, val_idx]
                
                # Train Gaussian Process on this fold
                gp = GaussianProcess(X_train, y_train, ktype, None, noise)
                
                # Make predictions on validation set
                y_pred, _ = gp.predict(X_val)
                
                # Compute MSE for this fold
                fold_error = np.mean((y_pred - y_val.T) ** 2)
                fold_errors.append(fold_error)
            
            # Average validation error over all folds
            errors[0, j] = np.mean(fold_errors)
            
        # Find the best noise parameter
        best_i, best_j = np.unravel_index(np.argmin(errors), errors.shape)
        best_noise = noise_vars[best_j]
        bestP = None  # Linear kernel has no parameter
        lowest_error = errors[best_i, best_j]
    else:
        # For RBF and polynomial kernels, vary both kernel parameters and noise
        for i, param in enumerate(paras):
            for j, noise in enumerate(noise_vars):
                fold_errors = []
                
                for train_idx, val_idx in kf.split(indices):
                    # Extract training and validation data for this fold
                    X_train, X_val = xTr[:, train_idx], xTr[:, val_idx]
                    y_train, y_val = y_std[:, train_idx], y_std[:, val_idx]
                    
                    try:
                        # Train Gaussian Process on this fold
                        gp = GaussianProcess(X_train, y_train, ktype, param, noise)
                        
                        # Make predictions on validation set
                        y_pred, _ = gp.predict(X_val)
                        
                        # Compute MSE for this fold
                        fold_error = np.mean((y_pred - y_val.T) ** 2)
                        fold_errors.append(fold_error)
                    except Exception as e:
                        print(f"Warning: Skipping param {param}, noise {noise} due to: {str(e)}")
                        # Assign a high error to this combination
                        fold_errors.append(float('inf'))
                
                # Average validation error over all folds
                errors[i, j] = np.mean(fold_errors)
        
        # Find the best parameter combination
        best_i, best_j = np.unravel_index(np.argmin(errors), errors.shape)
        best_noise = noise_vars[best_j]
        bestP = paras[best_i]
        lowest_error = errors[best_i, best_j]
    
    return best_noise, bestP, lowest_error, errors