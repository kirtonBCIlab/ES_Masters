import numpy as np

class SSVEPTransformer():
    def __init__(self, shape: tuple[int]) -> None:
        self._shape = shape
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ No need to fit. """
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """ Reshapes the data into the desired shape. """
        out = X.reshape((X.shape[0],) + self._shape)
        return out