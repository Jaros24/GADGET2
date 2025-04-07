'''
Collection of functions for curve fitting
'''

import numpy as np

def gaussian(x, amplitude, mu, sigma):
        return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))