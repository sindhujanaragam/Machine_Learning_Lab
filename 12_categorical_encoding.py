"""Practice 12: Label encode a categorical column."""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.DataFrame({'Name':['Ram','Sita','John','Ravi'], 'Department':['CSE','ECE','EEE','CSE']})
print('Original Dataset'); print(df)
encoder = LabelEncoder(); df['Department_Encoded'] = encoder.fit_transform(df['Department'])
print('\nEncoded Dataset'); print(df); print('\nCategory Mapping')
for category, value in zip(encoder.classes_, encoder.transform(encoder.classes_)): print(category, '->', value)
