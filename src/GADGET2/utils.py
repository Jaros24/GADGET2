import numpy as np
import time

def gaussian(x, amplitude, mu, sigma):
    """Standard function for curve fitting"""
    return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))

def vprint(message, verbose=False):
    """For debugging purposes. Prints message if verbose is True"""
    if verbose:
        print(message)
        time.sleep(0.1)