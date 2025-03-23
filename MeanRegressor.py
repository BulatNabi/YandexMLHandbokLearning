from sklearn.base import RegressorMixin
import numpy as np

class MeanRegressor(RegressorMixin):
    def __init__(self):
        self.mean = None

    def fit(self, X=None, y=None):

        if y is None:
            raise ValueError("y cannot be None")
        self.mean = np.mean(y)
        return self

    def predict(self, X=None):

        if self.mean is None:
            raise ValueError("Model is not fitted yet")
        if X is None:
            raise ValueError("X cannot be None")
        return np.full(X.shape[0], self.mean)


