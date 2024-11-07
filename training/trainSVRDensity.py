import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.svm import SVR


df = pd.read_csv("../fingerprinted_data/fingerprintedDensity.csv")

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
svr = SVR(kernel='rbf', C=1.0, epsilon=0.01)

svr.fit(x_train, y_train)

y_pred = svr.predict(x_test)

print(f"r2 score: {r2_score(y_test, y_pred)}")
print(f"mean absolute error: {mean_absolute_error(y_test, y_pred)}")