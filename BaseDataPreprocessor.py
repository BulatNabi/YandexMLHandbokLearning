from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler


class BaseDataPreprocessor(TransformerMixin):
    def __init__(self, needed_columns: Optional[List[str]] = None):
        """
        :param needed_columns: if not None select these columns from the dataframe
        """
        self.data = None
        self.needed_columns = needed_columns
        self.scaler = StandardScaler()

    def fit(self, data: pd.DataFrame, *args):
        """
        Prepares the class for future transformations
        :param data: pd.DataFrame with all available columns
        :return: self
        """
        # Your code here
        if self.needed_columns is not None:
            data = pd.DataFrame(data, columns=self.needed_columns)

        self.scaler.fit(data)
        return self

    def transform(self, data: pd.DataFrame) -> np.array:
        """
        Transforms features so that they can be fed into the regressors
        :param data: pd.DataFrame with all available columns
        :return: np.array with preprocessed features
        """
        # Your code here
        if self.needed_columns is not None:
            data = pd.DataFrame(data, columns=self.needed_columns)

        return self.scaler.transform(data)
