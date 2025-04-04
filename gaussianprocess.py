import numpy as np

class GaussianProcess:
    def __init__(self, X_train, y_train, ktype, kernel_param, noise):
        """
        Initialize and fit the Gaussian Process.
        
        Parameters:
            X_train: array-like, shape (n_train, n_features)
            y_train: array-like, shape (n_train,)
            ktype: str, kernel type: 'rbf', 'polynomial', or 'linear'
            kernel_param: kernel parameter (e.g., inverse kernel width for rbf, degree for polynomial)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.noise = noise

        # Set the kernel function based on ktype.
        if ktype == 'rbf':
            from rbfkernel import rbf_kernel
            self.kernel = lambda X, Y: rbf_kernel(X, Y, kpar=kernel_param)
        elif ktype == 'polynomial':
            from polynomialkernel import polynomial_kernel
            self.kernel = lambda X, Y: polynomial_kernel(X, Y, kpar=kernel_param)
        elif ktype == 'linear':
            from linearkernel import linear_kernel
            self.kernel = lambda X, Y: linear_kernel(X, Y)
        else:
            raise ValueError("Unsupported kernel type. Choose 'rbf', 'polynomial', or 'linear'.")
        
        self.fit()

    def fit(self):
        """
        Fit the GP to the training data by computing the kernel matrix and its Cholesky decomposition using matrix multiplications.
        This should also account for the noise parameter.
        """

        self.K = None
        self.alpha = None

        

    def predict(self, X_test):
        """
        Make predictions at test points.
        
        Parameters:
            X_test: array-like, shape (d,n)
        
        Returns:
            mean: array, shape (n,), predictive mean
            variance: array, shape (n,), predictive variance
        """
        # YOUR CODE HERE

        return mean, np.diag(variance)