import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from gaussianprocess import GaussianProcess
from standardize import standardize_targets, unstandardize_targets
from crossvalidate import crossvalidate
from negative_log_predictive_density import negative_log_predictive_density
import pickle
from sklearn.model_selection import train_test_split

def main():
    data = np.loadtxt("given.csv", delimiter=",")
    xTr = data[:, :-1].T  # training data should have dimensions (d,n)
    yTr = data[:, -1].reshape(1,-1) # targets should have dimensions (1,n)

    # Define hyperparameter grids - with very conservative ranges
    noise_vars = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    paras_dict = {
        'linear': [],  # Linear kernel has no hyperparameter
        'polynomial': [2, 3],  # The degree of the polynomial
        'rbf': [0.001, 0.01, 0.05, 0.1, 0.5]  # Inverse scale length (gamma)
    }
    
    best_kernel = None
    best_noise = None
    bestP = None
    lowest_error = float('inf')
    
    # Perform cross-validation for each kernel
    results = {}
    for ktype, paras in paras_dict.items():
        print(f"Cross-validating {ktype} kernel...")
        best_noise_k, bestP_k, lowest_error_k, error_grid = crossvalidate(xTr, yTr, ktype, noise_vars, paras)
        results[ktype] = (best_noise_k, bestP_k, lowest_error_k, error_grid)
        
        print(f"  Best noise: {best_noise_k}")
        print(f"  Best parameter: {bestP_k}")
        print(f"  Lowest RMSE: {np.sqrt(lowest_error_k)}")
        
        if lowest_error_k < lowest_error:
            best_kernel, best_noise, bestP, lowest_error = ktype, best_noise_k, bestP_k, lowest_error_k
    
    print("\nOverall Best Results:")
    print("Best Kernel:", best_kernel)
    print("Best Noise Variance:", best_noise)
    print("Best Kernel Parameter:", bestP)
    print("Lowest Average CV RMSE:", np.sqrt(lowest_error))

    # Combine best parameters into a dictionary
    best_parameters = {
        'kernel': best_kernel,
        'kpar': bestP,
        'noise': best_noise
    }

    # Save the best parameters
    pickle.dump(best_parameters, open('best_parameters.pickle', 'wb'))

    # Split data for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(xTr.T, yTr.T, test_size=0.2, random_state=42)
    X_train, X_test = X_train.T, X_test.T  # Transpose back to match expected dimensions
    y_train, y_test = y_train.T, y_test.T

    # Standardize targets
    y_train_std, y_mean, y_std = standardize_targets(y_train)
    
    # Train the GP with the best parameters
    gp = GaussianProcess(X_train, y_train_std, best_kernel, bestP, best_noise)
    
    # Make predictions
    y_pred_std, y_var = gp.predict(X_test)
    
    # Unstandardize predictions
    y_pred = unstandardize_targets(y_pred_std, y_mean, y_std)
    
    # Calculate RMSE on the original scale
    rmse = np.sqrt(mean_squared_error(y_test.T, y_pred))
    print(f"Test RMSE: {rmse}")
    
    # Also calculate RMSE on standardized scale for comparison
    y_test_std = (y_test - y_mean) / y_std
    std_rmse = np.sqrt(mean_squared_error(y_test_std.T, y_pred_std))
    print(f"Test RMSE (standardized scale): {std_rmse}")
    
    # Calculate NLPD for standardized predictions
    nlpd = negative_log_predictive_density(y_test_std.T, y_pred_std, y_var)
    print(f"Test NLPD: {nlpd}")
    
    # Visualize predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test.flatten(), y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'GP Regression with {best_kernel} kernel\nRMSE: {rmse:.2f}, NLPD: {nlpd:.2f}')
    plt.grid(True)
    plt.savefig('gp_predictions.png')
    
    # Visualize predictions differently - skip error bars for now
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test.flatten(), y_pred, alpha=0.7, label='Predictions')
    
    # Add a diagonal line for perfect predictions
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'GP Regression Predictions (RBF kernel)\nRMSE: {rmse:.2f}, NLPD: {nlpd:.2f}')
    plt.legend()
    plt.grid(True)
    plt.savefig('gp_predictions_scatter.png')
    
    # Try a simple plot of predictions vs. actual without error bars
    subset_size = min(50, len(y_test.flatten()))
    indices = np.random.choice(range(len(y_test.flatten())), subset_size, replace=False)
    
    y_test_subset = y_test.flatten()[indices]
    y_pred_subset = y_pred[indices]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(range(subset_size), y_test_subset, color='red', marker='x', label='Actual')
    plt.scatter(range(subset_size), y_pred_subset, color='blue', marker='o', label='Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Housing Price')
    plt.title('GP Predictions vs Actual Values')
    plt.legend()
    plt.grid(True)
    plt.savefig('gp_comparison.png')
    
    # If possible, visualize parameter sensitivity
    if best_kernel != 'linear':
        error_grid = results[best_kernel][3]
        plt.figure(figsize=(10, 6))
        plt.imshow(np.sqrt(error_grid), cmap='viridis', aspect='auto', interpolation='nearest')
        plt.colorbar(label='RMSE')
        
        # Set x and y ticks
        if best_kernel == 'rbf':
            params = paras_dict['rbf']
        else:
            params = paras_dict['polynomial']
            
        plt.yticks(range(len(params)), [str(p) for p in params])
        plt.xticks(range(len(noise_vars)), [str(n) for n in noise_vars])
        
        plt.xlabel('Noise Variance')
        plt.ylabel(f'{best_kernel.capitalize()} Kernel Parameter')
        plt.title(f'Cross-validation RMSE for {best_kernel.capitalize()} Kernel')
        plt.savefig('parameter_sensitivity.png')

if __name__ == "__main__":
    main()