"""
Utility functions for data strutures
"""

from fractions import Fraction

import attr

from h5preserve import Registry

ds_registry = Registry("disc_solver")


def _str_β_to_γ(β):
    """
    Convert β to γ where it appears as a string
    """
    return str(Fraction("5/4") - Fraction(β))


def get_fields(cls):
    """
    Get the list of field names from an attr.s class
    """
    return tuple(field.name for field in attr.fields(cls))
