"""Test cases for mesh setup functions."""

import unittest
from math import pi

import numpy as np
from numpy.testing import assert_allclose

from src.mesh import calc_areas



class MeshAreaTests(unittest.TestCase):
    """Runs tests on mesh area calculations and edge cases."""

    def test_planar_vertical(self):
        """assesses vertical planar face area"""
        bounds = (1.0, 10.0, 0.0)
        h = 1.0
        normal = (1.0, 0.0)
        curvature = 0
        depth = 1.0
        expected = np.r_[0.5, np.ones(8, float), 0.5]
        assert_allclose(calc_areas(bounds, h, normal, curvature, depth), expected, atol=1e-6)


    def test_planar_horizontal(self):
        """tests horizontal planar face area"""

        bounds = (1.0, 10.0, 0.0)
        h = 1.0
        normal = (0.0, 1.0)
        curvature = 0
        depth = 1.0
        expected = np.r_[0.5, np.ones(8, float), 0.5]
        assert_allclose(calc_areas(bounds, h, normal, curvature, depth), expected, atol=1e-6)


    def test_planar_neg_norm(self):
        """checks that negative normal direction does not mess with results."""

        bounds = (1.0, 10.0, 0.0)
        h = 1.0
        normal = (-1.0, 0.0)
        curvature = 0
        depth = 1.0
        expected = np.r_[0.5, np.ones(8, float), 0.5]
        assert_allclose(calc_areas(bounds, h, normal, curvature, depth), expected, atol=1e-6)


    def test_planar_neg_bounds(self):
        """check that negative bound does not affect results."""
        bounds = (-1.0, 8.0, 0.0)
        h = 1.0
        normal = (1.0, 0.0)
        curvature = 0
        depth = 1.0
        expected = np.r_[0.5, np.ones(8, float), 0.5]
        assert_allclose(calc_areas(bounds, h, normal, curvature, depth), expected, atol=1e-6)


    def test_planar_reverse_bounds(self):
        """check that s > e bounds does not affect results."""
        bounds = (10.0, 1.0, 0.0)
        h = 1.0
        normal = (1.0, 0.0)
        curvature = 0
        depth = 1.0
        expected = np.r_[0.5, np.ones(8, float), 0.5]
        assert_allclose(calc_areas(bounds, h, normal, curvature, depth), expected, atol=1e-6)


    def test_curved_horizontal(self):
        """check nominal area calculation."""

        bounds = (2.0, 6.0, 0.0)
        h = 1.0
        normal = (0.0, 1.0)
        curvature = 1
        expected = np.array([7.06858347057703, 18.849555921538, 25.132741228718, 31.415926535897, 18.064157758141])
        assert_allclose(calc_areas(bounds, h, normal, curvature), expected, atol=1e-6)


    def test_curved_horizontal_neg_radius(self):
        """radius cannot be negative."""

        bounds = (-1.0, 6.0, 0.0)
        h = 1.0
        normal = (0.0, 1.0)
        curvature = 1

        with self.assertRaises(ValueError):
            calc_areas(bounds, h, normal, curvature)


    def test_curved_vertical(self):
        """vertical, curved face area."""

        bounds = (-2.0, 2.0, 1.0)
        h = 1.0
        normal = (1.0, 0.0)
        curvature = 1

        expected = pi*np.array([1, 2, 2, 2, 1], float)
        assert_allclose(calc_areas(bounds, h, normal, curvature), expected)


    def test_curved_vertical_neg_radius(self):
        """radius cannot be negative."""

        bounds = (1.0, 3.0, -1.0)
        h = 1.0
        normal = (1.0, 0.0)
        curvature = 1

        with self.assertRaises(ValueError):
            calc_areas(bounds, h, normal, curvature)



unittest.main()
