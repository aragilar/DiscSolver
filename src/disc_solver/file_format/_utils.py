"""
Utility functions for data strutures
"""

from fractions import Fraction

from h5preserve import Registry

ds_registry = Registry("disc_solver")


def _str_β_to_γ(β):
    """
    Convert β to γ where it appears as a string
    """
    return str(Fraction("5/4") - Fraction(β))
