"""Practice 4: Candidate Elimination boundary demonstration.
This is a compact educational implementation for the EnjoySport example."""
import pandas as pd

data = pd.read_csv('EnjoySport.csv')
X = data.iloc[:, :-1].astype(str).values
y = data.iloc[:, -1].astype(str).str.lower().values
n = X.shape[1]
S = None
G = [['?' for _ in range(n)]]
for x, label in zip(X, y):
    if label == 'yes':
        if S is None: S = x.copy()
        else:
            for j in range(n):
                if S[j] != x[j]: S[j] = '?'
        G = [g for g in G if all(g[j] == '?' or g[j] == S[j] for j in range(n))]
    elif S is not None:
        new_G = []
        for g in G:
            if all(g[j] == '?' or g[j] == x[j] for j in range(n)):
                for j in range(n):
                    if S[j] != '?' and x[j] != S[j] and g[j] == '?':
                        candidate = g.copy(); candidate[j] = S[j]
                        if candidate not in new_G: new_G.append(candidate)
            else: new_G.append(g)
        G = new_G
print('Specific Boundary:'); print(S)
print('General Boundary:')
for g in G: print(g)
print('Note: This simplified version illustrates boundary updates for the supplied small dataset.')
