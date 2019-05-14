from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np


if __name__ == "__main__":
    df = pd.read_csv('data_banknote_authentication.csv')
    X = df.drop(columns=["class"])
    y = df["class"].values
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': np.arange(1, 25)}
    knn_gscv = GridSearchCV(knn, param_grid, cv=5, return_train_score=True)
    knn_gscv.fit(X, y)
    plt.plot(np.arange(1, 25), 'mean_test_score',
             data=knn_gscv.cv_results_, label="Test Score")
    plt.plot(np.arange(1, 25), 'mean_train_score',
             data=knn_gscv.cv_results_, color='green', label="Train Score")
    plt.title("Comparing Accuracy with K Values")
    plt.xlabel("n_neighbors")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
