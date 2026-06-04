"""Test cases for face perimeter calculations."""

import unittest
from math import pi

from src.util import calc_face_perimeter



class PerimeterTests(unittest.TestCase):
    """Unit tests for the calc_face_perimeter function."""

    def test_planar(self):
        """Planar face case."""

        bounds = (0.0, 1.0, 0.0)
        depth = 2.5
        normal = (1.0, 0.0)
        curvature = 0

        self.assertEqual(calc_face_perimeter(bounds, normal, curvature, depth), 7.0)


    def test_curved_vertical(self):
        """Vertical curved face."""

        bounds = (-1.0, 1.0, 3.0)
        normal = (1.0, 0.0)
        curvature = 1

        self.assertEqual(calc_face_perimeter(bounds, normal, curvature), 12*pi)


    def test_curved_horizontal(self):
        """Horizontal curved face."""

        bounds = (0.5, 1.5, 0.0)
        normal = (0.0, -1.0)
        curvature = 1

        self.assertEqual(calc_face_perimeter(bounds, normal, curvature), 4*pi)
