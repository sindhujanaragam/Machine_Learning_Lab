"""Practice 13: Random Forest classifier on the built-in Iris dataset."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
iris = load_iris(); X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train); pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, pred)); print('Classification Report:\n', classification_report(y_test, pred)); print('Confusion Matrix:\n', confusion_matrix(y_test, pred))
print('Feature Importance:')
for feature, importance in zip(iris.feature_names, model.feature_importances_): print(f'{feature}: {importance:.4f}')
