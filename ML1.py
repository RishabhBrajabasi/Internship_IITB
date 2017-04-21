from sklearn.datasets import make_regression

# simulate a dataset with 500 factors, but only 5 out of them are truely
# informative factors, all the rest 495 are noises. assume y is your response
# variable 'Sales', and X are your possible factors
X, y = make_regression(n_samples=1000, n_features=500, n_informative=5, noise=5)

X.shape
Out[273]: (1000, 500)
y.shape
Out[274]: (1000,)

from sklearn.feature_selection import f_regression
# regressing Sales on each of factor individually, get p-values
_, p_values = f_regression(X, y)
# select significant factors p < 0.05
mask = p_values < 0.05
X_informative = X[:, mask]

X_informative.shape
Out[286]: (1000, 38)