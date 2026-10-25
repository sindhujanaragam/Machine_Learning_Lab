"""Practice 11: One-hot encoding categorical columns."""
import pandas as pd
data = {'Name':['Ram','Sita','John','Ravi'], 'Gender':['Male','Female','Male','Male'], 'City':['Delhi','Mumbai','Delhi','Chennai']}
df = pd.DataFrame(data); print('Original Dataset'); print(df)
encoded = pd.get_dummies(df, columns=['Gender','City'], dtype=int)
print('\nOne-Hot Encoded Dataset'); print(encoded)
