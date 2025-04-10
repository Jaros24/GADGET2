import unittest
from GADGET2.energy_calibration import to_MeV, to_counts
import numpy as np

class TestEnergyCalibration(unittest.TestCase):
    
    def test_calib_point_1(self):
        self.assertAlmostEqual(to_MeV(156745), 0.806, places=3)
        self.assertAlmostEqual(to_counts(0.806), 156745, places=3)
    
    def test_calib_point_2(self):
        self.assertAlmostEqual(to_MeV(320842), 1.679, places=3)
        self.assertAlmostEqual(to_counts(1.679), 320842, places=3)    
    
    def test_midpoint(self):
        self.assertGreater(to_MeV(238793), 0.806)
        self.assertLess(to_MeV(238793), 1.679)
        
        self.assertGreater(to_counts(1.242), 156745)
        self.assertLess(to_counts(1.242), 320842)
        
    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            to_MeV("invalid")
        with self.assertRaises(TypeError):
            to_counts("invalid")
            
        with self.assertRaises(ValueError):
            to_MeV(-1)
        with self.assertRaises(ValueError):
            to_counts(-1)
    
    def test_array_input(self):
        # checks if the function can handle numpy arrays
        counts = np.array([156745, 320842])
        expected_MeV = np.array([0.806, 1.679])
        np.testing.assert_array_almost_equal(to_MeV(counts), expected_MeV, decimal=3)
        
        MeV = np.array([0.806, 1.679])
        expected_counts = np.array([156745, 320842])
        np.testing.assert_array_almost_equal(to_counts(MeV), expected_counts, decimal=3)
    
    def test_calibration_points(self):
        calib1 = (1, 100)
        calib2 = (2, 200)
        expected_MeV = np.array([1, 1.5, 2, 3])
        expected_counts = np.array([100, 150, 200, 300])
        
        np.testing.assert_array_equal(to_MeV(expected_counts, calib1, calib2), expected_MeV)
        np.testing.assert_array_equal(to_counts(expected_MeV, calib1, calib2), expected_counts)
    

if __name__ == '__main__':
    unittest.main()