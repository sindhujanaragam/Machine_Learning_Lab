"""Practice 9: Linear SVM for heart-disease classification using heart.csv."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv('heart.csv')
X, y = data.drop(columns=['target']), data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler(); X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)
model = SVC(kernel='linear', random_state=42).fit(X_train, y_train); pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, pred)); print('Confusion Matrix:', confusion_matrix(y_test, pred)); print(classification_report(y_test, pred))
