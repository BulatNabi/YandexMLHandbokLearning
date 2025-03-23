import numpy as np
from sklearn.linear_model import Ridge


class ExponentialLinearRegression(Ridge):

    def __init__(self,
                 alpha=1.0,
                 fit_intercept=True,
                 copy_X=True,
                 max_iter=None,
                 tol=1e-4,
                 solver="auto",
                 positive=False,
                 random_state=None, ):
        super().__init__(alpha=1.0,
                         fit_intercept=True,
                         copy_X=True,
                         max_iter=None,
                         tol=1e-4,
                         solver="auto",
                         positive=False,
                         random_state=None, )

    def fit(self, X, y):
        super().fit(X, np.log(y))
        return self

    def predict(self, X):
        return np.exp(super().predict(X))

    def get_params(self, deep=True):
        return super().get_params(deep=False)

    def set_params(self, **parameters):
        return super().set_params(**parameters)