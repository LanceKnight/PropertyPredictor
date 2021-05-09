import numpy as np
from sklearn.model_selection import StratifiedKFold

X = np.array([[1, 2], [2, 4], [3, 2], [4, 4],[5,6],[3,4]])
#-------------0  1  2  3 4
y = np.array([0, 0, 0, 1,1,1])
skf = StratifiedKFold(n_splits=3, shuffle=False)
skf.get_n_splits(X, y)

for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
