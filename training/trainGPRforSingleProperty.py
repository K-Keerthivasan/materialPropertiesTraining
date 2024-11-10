from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
from pickle import dump
import sys
import os

def get_filenames():
    """ Prompt the user to enter the input and output file names. """
    input_file_name = input("Enter the input file name (located in ../fingerprinted_data/): ")
    output_file_name = input("Enter the desired output file name (will be saved in ../models/): ")
    return input_file_name, output_file_name

def trainModel(input_file_name, output_file_name):
    # Construct file paths
    input_file = f"../fingerprinted_data/{input_file_name}"
    output_file = f"../models/{output_file_name}.pkl"

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load the data
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        sys.exit(1)

    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Setup the Gaussian Process model
    kernel = RBF(20.0, (1e-7, 1e5))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(x_train, y_train)

    # Predict and evaluate the model
    y_pred, sigma = gp.predict(x_test, return_std=True)
    print(f"r2 score: {r2_score(y_test, y_pred)}")
    print(f"mean absolute error: {mean_absolute_error(y_test, y_pred)}")

    # Export the model
    with open(output_file, "wb") as f:
        dump(gp, f, protocol=5)
    print(f"Model saved to {output_file}")

if __name__ == "__main__":
    input_file_name, output_file_name = get_filenames()
    trainModel(input_file_name, output_file_name)
