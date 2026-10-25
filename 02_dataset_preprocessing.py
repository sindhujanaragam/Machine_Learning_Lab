"""Practice 2: Load, inspect, clean, encode and split a dataset.
Uses student.csv in this folder; change TARGET to your target column if needed."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('student.csv')
print(df.head()); print('Shape:', df.shape); df.info(); print(df.describe(include='all'))
print('Missing values:', df.isnull().sum()); print('Duplicates:', df.duplicated().sum())
df['age'] = df['age'].fillna(df['age'].median())
df = df.dropna().drop_duplicates().copy()
for col in df.select_dtypes(include='object').columns:
    if col != 'purchased': df[col] = LabelEncoder().fit_transform(df[col].astype(str))
TARGET = 'purchased'
X, y = df.drop(columns=[TARGET]), df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print('Training samples:', len(X_train)); print('Testing samples:', len(X_test))
