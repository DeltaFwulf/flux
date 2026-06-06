"""Tests for material functions."""

import unittest
import numpy as np
from numpy.testing import assert_array_equal

from src.util import material_properties



class TestMaterials(unittest.TestCase):
    """tests material properties function."""

    def test_get_props_with_float(self):
        """function should accept a single float temp."""

        testonium = {'label':'test material',
                     'u':[100.0, 200.0],
                     'k':[1.0, 2.0],
                     'cp':[900.0, 1000.0],
                     'rho':[800.0, 900.0],
                     'emissivity':0.75,
                     'default_colour':'magenta'}

        expected = {'k':1.5, 'cp':950.0, 'rho':850.0, 'emissivity':0.75}
        self.assertDictEqual(material_properties(150.0, testonium), expected)


    def test_get_props_with_array(self):
        """function should also accept nd-arrays of temp."""

        testonium = {'label':'test material',
                     'u':[100.0, 200.0],
                     'k':[1.0, 2.0],
                     'cp':[900.0, 1000.0],
                     'rho':[800.0, 900.0],
                     'emissivity':0.75,
                     'default_colour':'magenta'}

        expected = {'k':np.array([[1.5, 1.5],[1.5, 1.5]]),
                    'cp':np.array([[950.0, 950.0],[950.0, 950.0]]),
                    'rho':np.array([[850.0, 850.0],[850.0, 850.0]]),
                    'emissivity':0.75}

        temp = np.array([[150.0, 150.0], [150.0, 150.0]])

        props = material_properties(temp, testonium)

        self.assertSetEqual(set(props.keys()), set(('k', 'cp', 'rho', 'emissivity')))

        assert_array_equal(props['k'], expected['k'])
        assert_array_equal(props['cp'], expected['cp'])
        assert_array_equal(props['rho'], expected['rho'])
        self.assertEqual(props['emissivity'], expected['emissivity'])
