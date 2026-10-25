"""Practice 10: Remove duplicates, evaluate a Decision Tree and run 5-fold CV."""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
iris = load_iris(); df = pd.DataFrame(iris.data, columns=iris.feature_names); df['target'] = iris.target
df = pd.concat([df, df.iloc[:5]], ignore_index=True)
print('Shape before removing duplicates:', df.shape); df = df.drop_duplicates(); print('Shape after removing duplicates:', df.shape)
X, y = df.iloc[:, :-1], df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
print('Model Accuracy:', accuracy_score(y_test, model.predict(X_test)))
scores = cross_val_score(model, X, y, cv=5); print('Cross Validation Scores:', scores); print('Average CV Accuracy:', scores.mean())
print('Bias: Error from overly simple assumptions.'); print('Variance: Sensitivity to changes in training data.')
