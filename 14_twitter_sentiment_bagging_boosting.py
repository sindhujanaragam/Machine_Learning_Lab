"""Practice 14: Twitter sentiment classification with Bagging and AdaBoost.
Expected CSV columns: Tweet, Sentiment. See README for dataset note."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('twitter_training.csv')
# Kaggle's common twitter_training.csv may have unnamed columns: ID, Entity, Sentiment, Tweet.
if not {'Tweet','Sentiment'}.issubset(df.columns) and df.shape[1] >= 4:
    df = df.iloc[:, [3, 2]].copy(); df.columns = ['Tweet', 'Sentiment']
if not {'Tweet','Sentiment'}.issubset(df.columns): raise ValueError('CSV must contain Tweet and Sentiment columns (or Kaggle four-column format).')
df = df[['Tweet','Sentiment']].dropna(); df['Tweet'] = df['Tweet'].astype(str)
X_train_text, X_test_text, y_train, y_test = train_test_split(df['Tweet'], df['Sentiment'], test_size=0.2, random_state=42, stratify=df['Sentiment'])
vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)
X_train = vectorizer.fit_transform(X_train_text); X_test = vectorizer.transform(X_test_text)
bagging = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=20), n_estimators=20, random_state=42, n_jobs=-1).fit(X_train, y_train)
bag_pred = bagging.predict(X_test); print('Bagging Accuracy:', accuracy_score(y_test, bag_pred)); print(classification_report(y_test, bag_pred, zero_division=0))
# AdaBoost is compatible with dense input; limit features to keep memory manageable.
X_train_dense, X_test_dense = X_train.toarray(), X_test.toarray()
boosting = AdaBoostClassifier(n_estimators=50, random_state=42).fit(X_train_dense, y_train)
boost_pred = boosting.predict(X_test_dense); print('AdaBoost Accuracy:', accuracy_score(y_test, boost_pred)); print(classification_report(y_test, boost_pred, zero_division=0))
