from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

import pandas as pd

from pickle import dump

import sys


def trainModel(input_file, output_file):

    df = pd.read_csv(input_file)

    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(x)

    # pca = PCA(n_components=0.95)

    # x_pca = pca.fit_transform(X_scaled)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-2, 1e2))
    kernel = RBF(20.0, (1e-7, 1e5))
    
    # kernel = Matern(length_scale=2.0, nu=0.5)

    # Create the Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    # gp = GaussianProcessRegressor()

    # param_grid = {'kernel': [RBF(length_scale=l) for l in [0.1 , 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]]}


                   
    # grid_search = GridSearchCV(gp, param_grid, cv=7)

    # grid_search.fit(x_train, y_train)

    # print("Best kernel:", grid_search.best_params_)

    # exit()

    # Fit the model to the training data
    gp.fit(x_train, y_train)

    # Make predictions on the test data
    y_pred, sigma = gp.predict(x_test, return_std=True)

    print(f"r2 score: {r2_score(y_test, y_pred)}")
    print(f"mean absolute error: {mean_absolute_error(y_test, y_pred)}")

    # export model
    with open(output_file + ".pkl", "wb") as f:
        dump(gp, f, protocol=5)


numArgs = len(sys.argv)

if(numArgs != 3):
    print("Usage: trainGPRSingleProperty.py INPUTFILE OUTPUTFILE")
    exit()

outputFile = sys.argv[2]
inputFile = sys.argv[1]

trainModel(inputFile, outputFile)

