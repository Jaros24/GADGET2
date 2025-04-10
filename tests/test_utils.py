import unittest
import numpy as np
from GADGET2.utils import gaussian, vprint

class TestGaussianFunction(unittest.TestCase):
    def test_gaussian_peak(self):
        # Test if the peak occurs at mu
        x = 0
        amplitude = 1
        mu = 0
        sigma = 1
        result = gaussian(x, amplitude, mu, sigma)
        self.assertAlmostEqual(result, amplitude, places=6)

    def test_gaussian_symmetry(self):
        # Test if the function is symmetric around mu
        x1 = -1
        x2 = 1
        amplitude = 1
        mu = 0
        sigma = 1
        result1 = gaussian(x1, amplitude, mu, sigma)
        result2 = gaussian(x2, amplitude, mu, sigma)
        self.assertAlmostEqual(result1, result2, places=6)

    def test_gaussian_zero_at_infinity(self):
        # Test if the function approaches zero as x moves far from mu
        x = 1e6
        amplitude = 1
        mu = 0
        sigma = 1
        result = gaussian(x, amplitude, mu, sigma)
        self.assertAlmostEqual(result, 0, places=6)

    def test_gaussian_negative_sigma(self):
        # Test if function returns same value for negative sigma
        x = 0
        amplitude = 1
        mu = 0
        sigma = -1
        result1 = gaussian(x, amplitude, mu, sigma)
        result2 = gaussian(x, amplitude, mu, abs(sigma))
        self.assertAlmostEqual(result1, result2, places=6)

class TestVPrint(unittest.TestCase):
    def test_verbose(self):
        verbose = False
        message = 'Testing'
        
        self.assertNoLogs(vprint(message, verbose))
        
        verbose = True
        self.assertLogs(vprint(message, verbose))

if __name__ == '__main__':
    unittest.main()