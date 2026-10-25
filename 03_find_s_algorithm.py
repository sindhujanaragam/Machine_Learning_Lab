"""Practice 3: FIND-S algorithm. Reads EnjoySport.csv from this folder."""
import pandas as pd

data = pd.read_csv('EnjoySport.csv')
features = data.columns[:-1]
hypothesis = None
for _, row in data.iterrows():
    if str(row.iloc[-1]).strip().lower() == 'yes':
        example = row.iloc[:-1].tolist()
        if hypothesis is None: hypothesis = example.copy()
        else:
            for i, value in enumerate(example):
                if hypothesis[i] != value: hypothesis[i] = '?'
print('Most Specific Hypothesis:')
print(hypothesis)
