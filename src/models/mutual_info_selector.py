import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif

class MutualInfoSelector(BaseEstimator, TransformerMixin):
    def __init__(self, keep_frac=0.6, random_state=42):
        """
        keep_frac: fraction of top MI features to keep (0 < keep_frac <= 1)
        """
        self.keep_frac = keep_frac
        self.random_state = random_state
        self.selected_cols = None

    def fit(self, X, y):
        # Convert to DataFrame (easier feature name handling)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Compute MI for each column
        mi = mutual_info_classif(
            X.values, y, 
            discrete_features=False, 
            random_state=self.random_state
        )
        mi_series = pd.Series(mi, index=X.columns)

        # Sort MI scores descending
        mi_sorted = mi_series.sort_values(ascending=False)

        # Select top fraction
        k = max(1, int(len(mi_sorted) * self.keep_frac))
        self.selected_cols = mi_sorted.index[:k].tolist()

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Select only the chosen columns
        return X[self.selected_cols]
