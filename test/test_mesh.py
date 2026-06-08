"""Mesh unit tests."""

import unittest
import numpy as np

from src.mesh import create_mesh, slice_regions

# TODO: break this down into the sub routines, rather than this high level test at first
#       (what does it prove?)


class MeshTests(unittest.TestCase):
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
                        'net_energy'})

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
