"""Test cases for common utility functions."""

import unittest

from src.util import calc_bc_relations


class EdgeRelationTests(unittest.TestCase):
    """Test cases for boundary condition linking."""

    def test_assign_relevant(self):
        """Each edge should be given all bcs that correspond to it."""

        edges = [0, 1, 2, 3, 4, 5]
        boundary_conditions = [{'edge':2},
                               {'edge':3},
                               {'edge':3},
                               {'edge':5},
                               {'edge':0}]

        expected = [[4], [], [0], [1, 2], [], [3]]
        self.assertSequenceEqual(calc_bc_relations(edges, boundary_conditions), expected)
