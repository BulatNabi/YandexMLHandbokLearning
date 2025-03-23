import math

import numpy as np
import pandas as pd
from typing import Optional, List

from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from BaseDataPreprocessor import BaseDataPreprocessor
from MSLE import root_mean_squared_logarithmic_error
from ExponentialLinearRegression import ExponentialLinearRegression
seed = 24





data = pd.read_csv("C:/Users/булат/Downloads/data (1).csv")
target_column = "\'Sale_Price\'"
np.random.seed(seed)
test_size = 0.2
data_train, data_test, Y_train, Y_test = train_test_split(
    data[data.columns.drop(target_column)],
    np.array(data[target_column]),
    test_size=test_size,
    random_state=seed)

continuous_columns = [key for key in data.keys() if data[key].dtype in ("int64", "float64")]
categorical_columns = [key for key in data.keys() if data[key].dtype == "object"]

continuous_columns.remove(target_column)

preprocessor = BaseDataPreprocessor(needed_columns=continuous_columns)

X_train = preprocessor.fit_transform(data_train)
X_test = preprocessor.transform(data_test)
X = np.concatenate((X_train, X_test), axis=0)
Y = np.concatenate((Y_train, Y_test), axis=0)


kf=KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scorer = make_scorer(root_mean_squared_logarithmic_error, greater_is_better=False)

alphas = np.logspace(-3, 3, num=7, base=10.)
param_grid = {
    'alpha': alphas
}
model = ExponentialLinearRegression()

gr = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring=rmsle_scorer,
    verbose=1,
    n_jobs=-1
)

gr.fit(X, Y)

print("Best parameters:", gr.best_params_)
print("Best score:", gr.best_score_)


