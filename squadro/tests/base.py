from unittest import TestCase

import numpy as np


class Base(TestCase):

    def assertEqualGeneral(self, first, second):
        if isinstance(first, np.ndarray):
            np.testing.assert_array_almost_equal(first, second, decimal=5)
        else:
            self.assertEqual(first, second)
