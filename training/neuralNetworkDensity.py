from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd


df = pd.read_csv('../fingerprinted_data/fingerprintedDensity.csv')


x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=1)

# scaler = StandardScaler()

# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

mlp_regressor = MLPRegressor(hidden_layer_sizes=(2,), random_state=1)

mlp_regressor.fit(x_train, y_train)

y_pred = mlp_regressor.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error: ", mse)
print("r2 score: ", r2)

