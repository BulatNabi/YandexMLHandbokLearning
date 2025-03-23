import math

import numpy as np
import pandas as pd
from typing import Optional, List

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from BaseDataPreprocessor import BaseDataPreprocessor
from MSLE import root_mean_squared_logarithmic_error
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

model = LinearRegression()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)


# Оцениваем качество модели с помощью MAE
mae = root_mean_squared_logarithmic_error(Y_test, Y_pred)
print(f"root_mean_squared_logarithmic_error:{mae}")


