"""Test cases for mesh setup functions."""

import unittest
from math import pi

import numpy as np
from numpy.testing import assert_allclose

from src.mesh import edge_area, volume



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
        assert_allclose(edge_area(bounds, h, normal, curvature, depth), expected, atol=1e-6)


    def test_planar_horizontal(self):
        """tests horizontal planar face area"""

        bounds = (1.0, 10.0, 0.0)
        h = 1.0
        normal = (0.0, 1.0)
        curvature = 0
        depth = 1.0
        expected = np.r_[0.5, np.ones(8, float), 0.5]
        assert_allclose(edge_area(bounds, h, normal, curvature, depth), expected, atol=1e-6)


    def test_planar_neg_norm(self):
        """checks that negative normal direction does not mess with results."""

        bounds = (1.0, 10.0, 0.0)
        h = 1.0
        normal = (-1.0, 0.0)
        curvature = 0
        depth = 1.0
        expected = np.r_[0.5, np.ones(8, float), 0.5]
        assert_allclose(edge_area(bounds, h, normal, curvature, depth), expected, atol=1e-6)


    def test_planar_neg_bounds(self):
        """check that negative bound does not affect results."""
        bounds = (-1.0, 8.0, 0.0)
        h = 1.0
        normal = (1.0, 0.0)
        curvature = 0
        depth = 1.0
        expected = np.r_[0.5, np.ones(8, float), 0.5]
        assert_allclose(edge_area(bounds, h, normal, curvature, depth), expected, atol=1e-6)


    def test_curved_horizontal(self):
        """check nominal area calculation."""

        bounds = (2.0, 6.0, 0.0)
        h = 1.0
        normal = (0.0, 1.0)
        curvature = 1
        expected = np.array([7.06858347057703,
                             18.849555921538,
                             25.132741228718,
                             31.415926535897,
                             18.064157758141])

        assert_allclose(edge_area(bounds, h, normal, curvature), expected, atol=1e-6)


    def test_curved_horizontal_reverse(self):
        """Areas should match elements in reverse order."""
        bounds = (6.0, 2.0, 0.0)
        h = 1.0
        normal = (0.0, 1.0)
        curvature = 1

        expected = np.array([18.064157758141,
                             31.415926535897,
                             25.132741228718,
                             18.849555921538,
                             7.06858347057703])

        assert_allclose(edge_area(bounds, h, normal, curvature), expected, atol=1e-6)


    def test_curved_horizontal_neg_radius(self):
        """radius cannot be negative."""

        bounds = (-1.0, 6.0, 0.0)
        h = 1.0
        normal = (0.0, 1.0)
        curvature = 1

        with self.assertRaises(ValueError):
            edge_area(bounds, h, normal, curvature)


    def test_curved_vertical(self):
        """vertical, curved face area."""

        bounds = (-2.0, 2.0, 1.0)
        h = 0.5
        normal = (1.0, 0.0)
        curvature = 1

        expected = 2*pi*h*np.r_[0.5, np.ones(7), 0.5]
        assert_allclose(edge_area(bounds, h, normal, curvature), expected)


    def test_curved_vertical_neg_radius(self):
        """radius cannot be negative."""

        bounds = (1.0, 3.0, -1.0)
        h = 1.0
        normal = (1.0, 0.0)
        curvature = 1

        with self.assertRaises(ValueError):
            edge_area(bounds, h, normal, curvature)


    def test_array_length(self):
        """The returned array should have same length as edge has points."""

        bounds = (0.111, 0.211, 0.0)
        h = 0.00625
        normal = (1.0, 0.0)
        curvature = 1

        self.assertEqual(edge_area(bounds, h, normal, curvature).size, 17)



class MeshVolTests(unittest.TestCase):
    """Check that volumes are calculated correctly."""


    def test_block(self):
        """check planar mesh elemental volumes."""

        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        dy = 0.25

        reg_x = [
            [{'bounds':(0, 4), 'length':1.0, 'type':'edge', 'direction':1, 'line':0}],
            [{'bounds':(0, 4), 'length':1.0, 'type':'internal', 'line_s':3, 'line_e':1}],
            [{'bounds':(0, 4), 'length':1.0, 'type':'internal', 'line_s':3, 'line_e':1}],
            [{'bounds':(0, 4), 'length':1.0, 'type':'internal', 'line_s':3, 'line_e':1}],
            [{'bounds':(0, 4), 'length':1.0, 'type':'edge', 'direction':-1, 'line':2}],
        ]

        vol = volume(reg_x, x, dy, curvature=0, depth=3.0)

        # corner
        self.assertEqual(vol[0, 0], 0.046875)
        # edge
        self.assertEqual(vol[0, 1], 0.09375)
        # internal
        self.assertEqual(vol[1, 1], 0.1875)
        # total volume
        self.assertEqual(np.sum(vol), 3.0)


    def test_cylinder(self):
        """check cylindrical mesh elemental volumes."""

        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        dy = 0.25

        reg_x = [
            [{'bounds':(0, 4), 'length':1.0, 'type':'edge', 'direction':1, 'line':0}],
            [{'bounds':(0, 4), 'length':1.0, 'type':'internal', 'line_s':3, 'line_e':1}],
            [{'bounds':(0, 4), 'length':1.0, 'type':'internal', 'line_s':3, 'line_e':1}],
            [{'bounds':(0, 4), 'length':1.0, 'type':'internal', 'line_s':3, 'line_e':1}],
            [{'bounds':(0, 4), 'length':1.0, 'type':'edge', 'direction':-1, 'line':2}],
        ]

        vol = volume(reg_x, x, dy, curvature=1)

        # corner
        self.assertEqual(vol[0, 0], 0.006135923151542565)
        # edges
        self.assertEqual(vol[0, 1], 0.01227184630308513)
        self.assertEqual(vol[1, 0], 0.04908738521234052)
        # internal
        self.assertEqual(vol[1, 1], 0.09817477042468104)
        # total volume
        self.assertEqual(np.sum(vol), 3.141592653589793)


    # def test_multi_region(self):
    #     """If a slice has multiple regions, left side excluded."""
    #     x = np.array([0.0, 0.25, 0.5, 0.75, 1.0], float)
    #     dy = 0.25

    #     # put a 1x1 hole in the 5x5 grid
    #     reg_x = [
    #         [{'bounds':(0, 4), 'length':1.0, 'type':'edge', 'direction':1, 'line':0}],
    #         [{'bounds':(0, 4), 'length':1.0, 'type':'internal', 'line_s':3, 'line_e':1}],
    #         [{'bounds':(0, 4), 'length':1.0, 'type':'internal', 'line_s':3, 'line_e':1}],
    #         [{'bounds':(0, 4), 'length':1.0, 'type':'internal', 'line_s':3, 'line_e':1}],
    #         [{'bounds':(0, 4), 'length':1.0, 'type':'edge', 'direction':-1, 'line':2}],
    #     ]
