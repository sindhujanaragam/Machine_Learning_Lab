"""Practice 1: Basic Python, NumPy, Pandas and Matplotlib commands."""
# Install dependencies once in your terminal: pip install notebook ipykernel pandas numpy matplotlib scikit-learn
print("Welcome to Machine Learning Lab")
a, b = 20, 10
print("Sum:", a + b)
for i in range(1, 6): print(i)
import math
print("Square root:", math.sqrt(64))
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print("Array values:", arr)
print("Adding 5:", arr + 5)
import pandas as pd
df = pd.DataFrame({'stuname':['raju','rama','sita'], 'stuno':[1,2,3], 'marks':[90,80,70]})
print(df)
import matplotlib.pyplot as plt
plt.bar([1,2,3,4,5], [10,20,25,30,40])
plt.title('Simple Bar Plot'); plt.xlabel('X-axis'); plt.ylabel('Y-axis'); plt.tight_layout(); plt.show()
