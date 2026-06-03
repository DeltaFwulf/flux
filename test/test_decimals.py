"""Contains tests for functions assessing floating point numbers."""

import unittest

from src.util import get_decimal_resolution

class ResolutionTests(unittest.TestCase):
    """Tests the get_decimal_resolution function."""

    def test_int(self):
        """ints have resolution 0."""
        self.assertEqual(get_decimal_resolution(1), 0)


    def test_negative_int(self):
        """negative should not change result."""
        self.assertEqual(get_decimal_resolution(-1), 0)


    def test_int_as_float(self):
        """integers expressed as float should still have resolution 0."""
        self.assertEqual(get_decimal_resolution(1.0), 0)


    def test_float(self):
        """returned value should match expected trailing digits."""
        self.assertEqual(get_decimal_resolution(1.123456), 6)


    def test_negative_float(self):
        """negative float should behave same as positive."""
        self.assertEqual(get_decimal_resolution(-1.123456), 6)
