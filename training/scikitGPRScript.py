import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate synthetic data (you would replace this with your polymer dataset)
# X represents polymer structure features, y represents polymer properties
X = np.random.rand(100, 1) * 10  # e.g., one structural feature
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])  # Polymer properties with some noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the kernel: RBF (Radial Basis Function) kernel with a constant term
kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-2, 1e2))

# Create the Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Fit the model to the training data
gp.fit(X_train, y_train)

# Make predictions on the test data
y_pred, sigma = gp.predict(X_test, return_std=True)

# Plot the results
plt.figure()
plt.scatter(X_train, y_train, color='red', label='Train data')
plt.scatter(X_test, y_test, color='blue', label='True test data')
plt.plot(X_test, y_pred, color='green', label='Predictions')
plt.fill_between(X_test.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, color='lightgreen', alpha=0.5, label='95% Confidence Interval')
plt.xlabel('Polymer Structure Feature')
plt.ylabel('Polymer Property')
plt.legend()
plt.show()

# Print kernel parameters and model score
print("Learned kernel: ", gp.kernel_)
print("Model score (R^2): ", gp.score(X_test, y_test))
