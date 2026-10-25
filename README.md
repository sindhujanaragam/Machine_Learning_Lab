# Machine Learning Lab Programs

This repository contains one Python file for each of the 14 practices in the uploaded Machine Learning Lab manual.

## Programs
1. Basic Python, NumPy, Pandas and Matplotlib
2. Dataset loading and preprocessing
3. FIND-S algorithm
4. Candidate Elimination boundary demonstration
5. Binary classification using Logistic Regression
6. KNN on Iris dataset
7. Linear Regression for house prices
8. Logistic Regression with feature scaling
9. Linear SVM for heart disease
10. Duplicate removal, bias/variance concepts and cross-validation
11. One-hot encoding
12. Label/categorical encoding
13. Random Forest classifier
14. Twitter sentiment analysis with Bagging and AdaBoost

## Run
Install Python 3.10+ and dependencies:
```bash
pip install -r requirements.txt
```
Run a program from this folder, for example:
```bash
python 06_knn_iris.py
```

## Dataset notes
Small illustrative CSV files are included for Practices 2, 3, 5, 7 and 9 so the programs have an example input. These are demo data, not official benchmark datasets, and model metrics can differ from the manual. `twitter_training.csv` is not bundled because the lab calls for downloading a dataset from Kaggle. For Practice 14, provide a CSV with `Tweet` and `Sentiment` columns, or the common four-column Kaggle Twitter training format. The heart and social-ad CSV files are tiny demonstrations, not for medical or real-world decision-making.

## Notes
- The source manual's dataset references and examples are retained where practical; scripts include a few compatibility/indentation fixes so they can run as Python files.
- Practice 4 is an educational, simplified Candidate Elimination implementation, not a fully general-purpose version-space library.
