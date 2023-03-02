
import os
from sklearn import datasets

import numpy as np
import pandas as pd

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")


def make_forge():
    # a carefully hand-designed dataset lol
    X, y = datasets.make_blobs(centers=2, random_state=4, n_samples=30)
    y[np.array([7, 27])] = 0
    mask = np.ones(len(X), dtype=np.bool)
    mask[np.array([0, 1, 5, 26])] = 0
    X, y = X[mask], y[mask]
    return X, y


def make_wave(n_samples=100):
    rnd = np.random.RandomState(42)
    x = rnd.uniform(-3, 3, size=n_samples)
    y_no_noise = (np.sin(4 * x) + x)
    y = (y_no_noise + rnd.normal(size=len(x))) / 2
    return x.reshape(-1, 1), y


def load_boston(*, return_X_y=False):
    import csv
    from sklearn.utils import Bunch
    filename = os.path.join(DATA_FOLDER, 'boston_house_prices.csv')
    with open(filename) as f:
        data_file = csv.reader(f)

        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,))
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.float64)

    if return_X_y:
        return data, target

    return Bunch(
        data=data,
        target=target,
        # last column is target value
        feature_names=feature_names[:-1],
        DESCR='',
        filename=filename,
    )


def load_extended_boston():
    from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

    boston = load_boston()
    X = boston.data

    X = MinMaxScaler().fit_transform(boston.data)
    X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
    return X, boston.target
