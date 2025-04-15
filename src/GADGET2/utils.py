import numpy as np
import sys
import os

def gaussian(x, amplitude, mu, sigma):
    """Standard function for curve fitting"""
    return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout