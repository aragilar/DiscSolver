
# -*- coding: utf-8 -*-

import unittest

from disc_solver import main

class MainTest(unittest.TestCase):
    def test_main(self):
        import matplotlib.pyplot as plt
        plt.ion()
        main()
        # pass only if main does not raise an exception
        self.assertTrue(True)
