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
        self.ktype = ktype
        self.kernel_param = kernel_param

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
        Fit the GP to the training data by computing the kernel matrix and its Cholesky decomposition.
        This should also account for the noise parameter.
        """
        # Compute the kernel matrix K
        self.K = self.kernel(self.X_train, self.X_train)
        
        # Add noise to the diagonal of K
        n = self.K.shape[0]
        
        # Ensure minimum noise level for numerical stability
        effective_noise = max(self.noise, 1e-8)
        
        # Add jitter to diagonal for numerical stability
        jitter = 1e-8
        self.K_noisy = self.K + (effective_noise + jitter) * np.eye(n)
        
        # Try Cholesky decomposition with increasing jitter if needed
        max_tries = 10
        current_jitter = jitter
        success = False
        
        for i in range(max_tries):
            try:
                # Compute Cholesky decomposition of K_noisy = L * L.T
                self.L = np.linalg.cholesky(self.K_noisy)
                
                # Compute alpha = K^{-1} * y using the Cholesky decomposition
                # First solve L * v = y for v
                v = np.linalg.solve(self.L, self.y_train.T)
                
                # Then solve L.T * alpha = v for alpha
                self.alpha = np.linalg.solve(self.L.T, v)
                
                # If we get here, the decomposition was successful
                success = True
                break
                
            except np.linalg.LinAlgError:
                # If Cholesky decomposition fails, add more jitter and try again
                current_jitter *= 10
                self.K_noisy = self.K + (effective_noise + current_jitter) * np.eye(n)
        
        if not success:
            # If all attempts failed, try a direct approach with regularization
            reg_value = effective_noise + current_jitter * 10
            self.K_noisy = self.K + reg_value * np.eye(n)
            
            # Use pseudoinverse as a fallback
            K_inv = np.linalg.pinv(self.K_noisy)
            self.alpha = K_inv @ self.y_train.T
            
            # For later variance calculations, approximate L
            self.L = np.linalg.cholesky(self.K_noisy + reg_value * np.eye(n))

    def predict(self, X_test):
        """
        Make predictions at test points.
        
        Parameters:
            X_test: array-like, shape (d,n)
        
        Returns:
            mean: array, shape (n,), predictive mean
            variance: array, shape (n,), predictive variance
        """
        try:
            # Compute the kernel between test and training points
            K_star = self.kernel(self.X_train, X_test)
            
            # Compute the predictive mean
            mean = K_star.T @ self.alpha
            
            # Compute the predictive variance
            # First solve L * v = K_star for v
            v = np.linalg.solve(self.L, K_star)
            
            # Compute the kernel for test points
            K_test_test = self.kernel(X_test, X_test)
            
            # Compute the variance
            variance = K_test_test - v.T @ v
            
            # Ensure all variances are positive
            variance = np.maximum(np.diag(variance), 1e-8)
            
            return mean, variance
            
        except Exception as e:
            # Fallback predictions if there are numerical issues
            print(f"Warning: Using fallback prediction method due to: {str(e)}")
            K_star = self.kernel(self.X_train, X_test)
            mean = K_star.T @ self.alpha
            
            # Simple variance estimate
            variance = np.ones(X_test.shape[1]) * self.noise
            
            return mean, variance