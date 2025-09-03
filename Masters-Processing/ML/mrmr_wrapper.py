import pandas as pd
import numpy as np
from mrmr import mrmr_classif

class MRMRTransformer:
    def __init__(self, k_features):
        self.k_features = k_features
        self.selected_features = None
        self.column_names = None
    
    def fit(self, X, y):
        # Convert to DataFrame if not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Reset indices to avoid alignment issues
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        
        self.column_names = X.columns.tolist()
        try:
            self.selected_features = mrmr_classif(X, y, K=self.k_features)
            print("Got MRMR features")
        except:
            # Fallback to random features if MRMR fails
            self.selected_features = np.random.choice(X.columns, size=min(self.k_features, len(X.columns)), replace=False)
            print("MRMR failed, selected random features instead.")
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.column_names)
        return X[self.selected_features]