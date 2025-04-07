import unittest
from GADGET2.test import test_func

class TestTestFunc(unittest.TestCase):
    def test_returns_true(self):
        self.assertTrue(test_func())

if __name__ == '__main__':
    unittest.main()