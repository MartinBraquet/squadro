from numbers import Number
from unittest import TestCase

import numpy as np


class Base(TestCase):

    def assertEqualGeneral(self, first, second):
        if isinstance(first, np.ndarray):
            np.testing.assert_array_almost_equal(first, second, decimal=5)
        elif isinstance(first, Number):
            self.assertAlmostEqual(first, second, places=5)
        else:
            self.assertEqual(first, second)

    @staticmethod
    def _check_keys(first, second):
        missing_second = list(set(first) - set(second))
        missing_first = list(set(second) - set(first))
        if missing_first or missing_second:
            raise ValueError(f"Keys {missing_first=} and {missing_second=} are missing")

    def assertAlmostEqualCustom(self, first, second, rel_tol=1e-5):
        """
        If a dictionary value is a number, then check that the numbers are almost equal, otherwise check if values are
        exactly equal
        Note: does not currently try converting strings to digits and comparing them. Does not care about ordering of
        keys in dictionaries
        Just returns true or false
        """
        if type(first) != type(second):
            raise ValueError(f"Data {first} and {second} are not the same type")
        if isinstance(first, dict):
            self._check_keys(list(first), list(second))
            for k in first.keys():
                self.assertAlmostEqualCustom(first[k], second[k], rel_tol)
        elif isinstance(first, list):
            if len(first) != len(second):
                raise ValueError(f"Lists {first} and {second} have different lengths")
            for v, v2 in zip(first, second):
                self.assertAlmostEqualCustom(v, v2, rel_tol)
        # elif isinstance(first, Number):
        # if not isclose(first, d2, rel_tol=rel_tol):
        #     raise ValueError(f"Numbers {first} and {d2} are not almost equal")
        elif isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
            np.testing.assert_array_almost_equal(first, second, decimal=rel_tol)
        elif isinstance(first, Number) and isinstance(second, Number):
            self.assertAlmostEqual(first, second, delta=rel_tol * abs(first) if isinstance(first,
                                                                                           float) else rel_tol)
            # if first != d2:
            #     raise ValueError(f"Values {first} and {d2} are not the same")
        else:
            self.assertEqual(first, second)
