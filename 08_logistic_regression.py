"""Practice 8: Logistic Regression with scaling on Social_Network_Ads.csv."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv('Social_Network_Ads.csv'); X, y = data[['Age','EstimatedSalary']], data['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler(); X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)
model = LogisticRegression(random_state=42, max_iter=1000).fit(X_train, y_train); pred = model.predict(X_test)
print('Actual	Predicted')
for actual, predicted in zip(y_test, pred): print(actual, '	', predicted)
print('Accuracy Score:', accuracy_score(y_test, pred)); print('Confusion Matrix:', confusion_matrix(y_test, pred)); print(classification_report(y_test, pred))
