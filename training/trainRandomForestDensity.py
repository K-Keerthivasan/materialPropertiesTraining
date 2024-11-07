import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("../fingerprinted_data/fingerprinted_Glass_transition_temperature.csv")

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=1000, random_state=42)

rf.fit(x_train, y_train)

# Make predictions on the test data
y_pred = rf.predict(x_test)

print(f"r2 score: {r2_score(y_test, y_pred)}")
print(f"mean absolute error: {mean_absolute_error(y_test, y_pred)}")