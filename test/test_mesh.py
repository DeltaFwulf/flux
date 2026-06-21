"""Test cases for mesh setup functions."""

import unittest
from math import pi

import numpy as np
from numpy.testing import assert_allclose

from src.mesh import edge_area, volume, grid_resolution, calc_face_perimeter, slice_regions, create_mesh

# TODO: add mass test cases



class MeshStructureTests(unittest.TestCase):
    """Tests to validate mesh data structure."""

    def test_structure(self):
        """Ensures all fields are present when mesh is built."""

        # pass the definition dictionary to the function
        mdef = {}
        mdef.update({'label':'test mesh'})
        mdef.update({'dx': 0.1})
        mdef.update({'dy': 0.1})
        mdef.update({'u0': 100.0})
        mdef.update({'material':'testonium'})
        mdef.update({'curvature':0})
        mdef.update({'depth':1.0})

        mdef.update({'lines':
                    [[[0.0, 0.0], [1.0, 0.0]],
                    [[1.0, 0.0], [1.0, 1.0]],
                    [[1.0, 1.0], [0.0, 1.0]],
                    [[0.0, 1.0], [0.0, 0.0]]]})

        mdef.update({'boundary_conditions':
                    [{'edge': 0, 'mode': 'neumann', 'value': 0.0},
                     {'edge': 1, 'mode': 'neumann', 'value': 0.0},
                     {'edge': 2, 'mode': 'neumann', 'value': 0.0},
                     {'edge': 3, 'mode': 'neumann', 'value': 0.0}]})

        testonium = {'label':'test material',
                     'u':[100.0, 200.0],
                     'k':[1.0, 2.0],
                     'cp':[900.0, 1000.0],
                     'rho':[800.0, 900.0],
                     'emissivity':0.75,
                     'default_colour':'magenta'}

        # does the mesh have all the expected fields, and are they the correct type?
        testmesh = create_mesh(mesh_def=mdef, force_finer=True, material=testonium)

        expected = set({'label',
                        'dx', 
                        'dy',
                        'depth',
                        'curvature',
                        'lines',
                        'boundary_conditions',
                        'i',
                        'j',
                        'x',
                        'y',
                        'material',
                        'k',
                        'rho',
                        'cp',
                        'emissivity',
                        'regions_x',
                        'regions_y',
                        'edges',
                        'edge_bcs',
                        'edge_powers',
                        'edge_powers_latest',
                        'u',
                        'u_prev',
                        'u_latest',
                        'mass',
                        'enthalpy'})

        self.assertSetEqual(set(testmesh.keys()), expected)


    def test_edge_region_x(self):
        """Should detect edges correctly in x."""

        direction = 'x'
        xp = np.arange(0.0, 1.1, 0.1)
        xn = 0.0
        lines = [[[0.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 1.0]],
                [[1.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 0.0]]]

        expected = [{'bounds':(0, 10), 'length':1.0, 'type':'edge', 'line':0, 'direction':1}]

        self.assertSequenceEqual(slice_regions(direction, xp, xn, lines), expected)


    def test_edge_region_y(self):
        """Should detect edges correctly in x."""

        direction = 'y'
        xp = np.arange(0.0, 1.4, 0.1)
        xn = 1.0
        lines = [[[0.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 1.3]],
                [[1.0, 1.3], [0.0, 1.3]],
                [[0.0, 1.3], [0.0, 0.0]]]

        expected = [{'bounds':(0, 13), 'length':1.3, 'type':'edge', 'line':1, 'direction':-1}]

        self.assertSequenceEqual(slice_regions(direction, xp, xn, lines), expected)


    def test_internal_region_x(self):
        """If not on edge, should return an internal region."""

        direction = 'x'
        xp = np.arange(0.0, 1.1, 0.1)
        xn = 0.3
        lines = [[[0.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 1.0]],
                [[1.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 0.0]]]

        expected = [{'bounds':(0, 10), 'length':1.0, 'type':'internal', 'line_s':3, 'line_e':1}]

        self.assertSequenceEqual(slice_regions(direction, xp, xn, lines), expected)


    def test_internal_region_y(self):
        """If not on edge, should return an internal region."""

        direction = 'y'
        xp = np.arange(0.0, 1.1, 0.1)
        xn = 0.2
        lines = [[[0.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 1.0]],
                [[1.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 0.0]]]

        expected = [{'bounds':(0, 10), 'length':1.0, 'type':'internal', 'line_s':0, 'line_e':2}]

        self.assertSequenceEqual(slice_regions(direction, xp, xn, lines), expected)


    def test_mass_planar(self):
        """Mass equals initial mean density*volume"""
        pass


    def test_mass_curved(self):
        """Mass = 2*pi*CSA"""
        pass



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
        dy = 0.5

        reg_x = [
            [{'bounds':(0, 4), 'length':1.0, 'type':'edge', 'direction':1, 'line':0}],
            [{'bounds':(0, 4), 'length':1.0, 'type':'internal', 'line_s':3, 'line_e':1}],
            [{'bounds':(0, 4), 'length':1.0, 'type':'internal', 'line_s':3, 'line_e':1}],
            [{'bounds':(0, 4), 'length':1.0, 'type':'internal', 'line_s':3, 'line_e':1}],
            [{'bounds':(0, 4), 'length':1.0, 'type':'edge', 'direction':-1, 'line':2}],
        ]

        vol = volume(reg_x, x, dy, curvature=1)

        # corner
        self.assertEqual(vol[0, 0], 0.01227184630308513)
        # edges
        self.assertEqual(vol[0, 1], 0.02454369260617026)
        self.assertEqual(vol[1, 0], 0.09817477042468104)
        # internal
        self.assertEqual(vol[1, 1], 0.19634954084936207)
        # total volume
        self.assertEqual(np.sum(vol), 6.283185307179586)


    def test_multi_region(self):
        """If a slice has multiple regions, left side excluded."""
        x = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], float)
        dy = 1.0

        # internal region follows an edge region
        reg_x = [
            [{'bounds':(0, 3), 'length':0.6, 'type':'edge', 'direction':1, 'line':0},
             {'bounds':(3, 5), 'length':0.4, 'type':'internal', 'line_s':3, 'line_e':1}
            ]]

        vol = volume(reg_x, x, dy, curvature=0, depth=3.0)
        assert_allclose(vol, np.array([[0.15, 0.3, 0.3, 0.45, 0.6, 0.3]]).transpose(), atol=1e-6)



class MeshResolutionTests(unittest.TestCase):
    """Tests for mesh resolution calculations."""

    def test_res_forcefiner(self):
        """Grid tiles mesh and is at least as fine as input resolution."""

        lines = [[[0.0, 0.0], [1.5, 0.0]],
                 [[1.5, 0.0], [1.5, 1.5]],
                 [[1.5, 1.5], [0.0, 1.5]],
                 [[0.0, 1.5], [0.0, 0.0]]]

        res = grid_resolution(lines, 0.2, 0.2, force_finer=True)

        self.assertEqual(res['dx'], 0.1875)
        self.assertEqual(res['dy'], 0.1875)


    def test_res_already_fits(self):
        """If the input resolution fits, don't alter it."""

        lines = [[[0.0, 0.0], [1.5, 0.0]],
                 [[1.5, 0.0], [1.5, 1.5]],
                 [[1.5, 1.5], [0.0, 1.5]],
                 [[0.0, 1.5], [0.0, 0.0]]]

        res = grid_resolution(lines, 0.3, 0.3, force_finer=True)

        self.assertEqual(res['dx'], 0.3)
        self.assertEqual(res['dy'], 0.3)


    def test_min_elements(self):
        """At least (min_elements + 1) points should be present at narrowest width."""

        lines = [[[0.0, 0.0], [1.5, 0.0]],
                 [[1.5, 0.0], [1.5, 1.5]],
                 [[1.5, 1.5], [0.0, 1.5]],
                 [[0.0, 1.5], [0.0, 0.0]]]

        res = grid_resolution(lines, 0.3, 0.3, force_finer=True, min_elements=10)

        self.assertEqual(res['dx'], 0.15)
        self.assertEqual(res['dy'], 0.15)



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
