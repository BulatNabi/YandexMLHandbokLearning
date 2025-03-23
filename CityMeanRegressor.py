from sklearn.base import RegressorMixin
import numpy as np
class CityMeanRegressor(RegressorMixin):
    def fit(self, X=None, y=None):
        cities = set(X)
        self.means = {}
        for city in cities:
            self.means[city] = np.mean(y[X == city])
        self.is_fitted_ = True
        return self
    def predict(self, X=None):
        return np.array([self.means[city] for city in X])