import unittest
import numpy as np
from GADGET2.EnergySpectrum import plot_1d_hist, show_cut, plot_spectrum, quick_fit_gaussian

import matplotlib.pyplot as plt
np.random.seed(1)  # For reproducibility

class TestEnergySpectrum(unittest.TestCase):

    def setUp(self):
        # Create a sample dataset for testing
        self.dataset = np.random.normal(loc=50, scale=10, size=10000)
        self.vlines = [30, 70]
        self.num_bins = 20
        self.units = "Integrated Charge (adc counts)"
        self.fig_name = "Energy Spectrum"

    def test_plot_1d_hist(self):
        # Test if plot_1d_hist runs without errors
        try:
            plot_1d_hist(self.dataset, self.fig_name, self.units, self.num_bins, self.vlines)
            plt.close()
        except Exception as e:
            self.fail(f"plot_1d_hist raised an exception: {e}")

    def test_show_cut(self):
        # Test if show_cut returns the correct count within the cut range
        count = show_cut(self.dataset, self.vlines, self.fig_name, self.units, self.num_bins)
        expected_count = len(self.dataset[(self.dataset >= self.vlines[0]) & (self.dataset <= self.vlines[1])])
        self.assertEqual(count, expected_count)

    def test_plot_spectrum(self):
        # Test if plot_spectrum runs without errors
        try:
            plot_spectrum(self.dataset, self.fig_name, self.units, self.num_bins, self.vlines)
            plt.close()
        except Exception as e:
            self.fail(f"plot_spectrum raised an exception: {e}")

    def test_quick_fit_gaussian(self):
        # Test if quick_fit_gaussian runs without errors
        try:
            quick_fit_gaussian(self.dataset, self.units, self.num_bins, self.vlines)
            plt.close()
        except Exception as e:
            self.fail(f"quick_fit_gaussian raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()