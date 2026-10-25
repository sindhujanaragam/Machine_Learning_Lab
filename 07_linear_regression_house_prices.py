"""Practice 7: Linear Regression using HousePrice.csv."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('HousePrice.csv')
X, y = data[['Area']], data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Actual Price	Predicted Price')
for actual, predicted in zip(y_test, y_pred): print(actual, '	', round(predicted, 2))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred)); print('R2 Score:', r2_score(y_test, y_pred))
