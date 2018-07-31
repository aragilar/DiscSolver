
# -*- coding: utf-8 -*-

from math import pi
import unittest

from disc_solver.utils import cot


class CotTest(unittest.TestCase):
    precision = 1e-15

    def test_pi(self):
        self.assertEqual(float('-inf'), cot(pi))

    def test_pi_div_2(self):
        self.assertEqual(0, cot(pi/2))

    def test_zero(self):
        self.assertEqual(float('inf'), cot(0))

    def test_pi_div_4(self):
        self.assertAlmostEqual(1, cot(pi/4), delta=self.precision)
