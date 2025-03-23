from scipy.stats import mode
from sklearn.base import ClassifierMixin
import numpy as np


class MostFrequentClassifier(ClassifierMixin):
    def __init__(self):
        self.most_frequent = None

    def fit(self, X=None, y=None):

        if y is None:
            raise ValueError("y cannot be None")
        self.most_frequent = mode(y)[0]
        return self

    def predict(self, X=None):

        if self.most_frequent is None:
            raise ValueError("Model is not fitted yet")
        if X is None:
            raise ValueError("X cannot be None")
        return np.full(X.shape[0], self.most_frequent)