import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from gaussianprocess import GaussianProcess
from standardize import standardize_targets, unstandardize_targets
from crossvalidate import crossvalidate
from negative_log_pred_density import negative_log_predictive_density
import pickle

def main():
    data = np.loadtxt("GPs/given.csv", delimiter=",")
    xTr = data[:, :-1].T  #training data should have dimensions (d,n)
    yTr = data[:, -1].reshape(1,-1) #targets should have dimensions (1,n)

    #define hyperparameter grids — you should experiment beyond these default values,
    #including different kernels and kernel parameter and noise values;
    #for numerical stability purposes, do not set the noise less than 1e-6

    noise_vars = [11]
    paras_dict = {
        'linear': [],  #the linear kernel does not have a hyperparameter; you should update the list accordingly
        'polynomial': [],  #the degree of the polynomial
        'rbf': [7]  #inverse scale length
    }
    
    best_kernel = None
    best_noise = None
    bestP = None
    lowest_error = float('inf')
    
    #perform x-fold cross-validation (choose # of folds in your implementation)
    results = {}
    for ktype, paras in paras_dict.items():
        best_noise_k, bestP_k, lowest_error_k, error_grid = crossvalidate(xTr, yTr, ktype, noise_vars, paras)
        results[ktype] = (best_noise_k, bestP_k, lowest_error_k, error_grid)
        
        if lowest_error_k < lowest_error:
            best_kernel, best_noise, bestP, lowest_error = ktype, best_noise_k, bestP_k, lowest_error_k
    
    print("Best Kernel:", best_kernel)
    print("Best Noise Variance:", best_noise)
    print("Best Kernel Parameter:", bestP)
    print("Lowest Average x-Fold Cross-Validation Root Mean Squared Error:", np.sqrt(lowest_error))

    #combine best parameters into a dictionary
    best_parameters = {
        'kernel': best_kernel,
        'kpar': bestP,
        'noise': best_noise
    }

    #save the best parameters — MAKE SURE TO INCLUDE THIS IN YOUR SUBMISSION
    pickle.dump(best_parameters, open('best_parameters.pickle', 'wb'))

    # YOUR CODE HERE


    #Things you may want to implement in main:
    # - creating plots to visualize the accuracy of your predictions
    # - compute NLPD for your standardized training points to make sure your function is reasonable; your
    #   function will be tested using the (hidden) test set

if __name__ == "__main__":
    main()
