"""Practice 6: KNN classification on the built-in Iris dataset."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris = load_iris(); X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = KNeighborsClassifier(n_neighbors=3); model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Actual  Predicted')
correct = wrong = 0
for actual, pred in zip(y_test, y_pred):
    print(actual, '     ', pred)
    if actual == pred: correct += 1
    else: wrong += 1
print('Correct Predictions:', correct); print('Wrong Predictions:', wrong); print('Accuracy:', accuracy_score(y_test, y_pred))
