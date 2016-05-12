# -*- coding: utf-8 -*-
"""
Utility function and classes for solver associated code
"""
from numpy import (
    any as np_any,
    diff,
    abs as np_abs,
)

from scikits.odes.sundials.cvode import StatusEnum


def ode_error_handler(error_code, module, func, msg, user_data):
    """ drop all CVODE messages """
    # pylint: disable=unused-argument
    pass


def gen_sonic_point_rootfn(c_s):
    """
    Finds acoustic sonic point
    """
    def rootfn(θ, params, out):
        """
        root function to find acoustic sonic point
        """
        # pylint: disable=unused-argument
        out[0] = c_s - params[5]
        return 0
    return rootfn


def validate_solution(soln):
    """
    Check that the solution returned is valid, even if the ode solver returned
    success
    """
    if soln.flag != StatusEnum.SUCCESS:
        return soln, False
    v_θ = soln.solution[:, 5]
    ρ = soln.solution[:, 6]
    if np_any(v_θ < 0):
        return soln, False
    if np_any(diff(v_θ) < 0):
        return soln, False
    if np_any(ρ < 0):
        return soln, False
    return soln, True


def onroot_continue(*args):
    """
    Always continue after finding root
    """
    # pylint: disable=unused-argument
    return 0


def onroot_stop(*args):
    """
    Always stop after finding root
    """
    # pylint: disable=unused-argument
    return 1


def ontstop_continue(*args):
    """
    Always continue after finding tstop
    """
    # pylint: disable=unused-argument
    return 0


def ontstop_stop(*args):
    """
    Always stop after finding tstop
    """
    # pylint: disable=unused-argument
    return 1


def closest(a, max_val):
    """
    Return closest value in a to max_val
    """
    return a[closest_index(a, max_val)]


def closest_less_than(a, max_val):
    """
    Return closest value in a to max_val which is less than max_val
    """
    return a[closest_less_than_index(a, max_val)]


def closest_index(a, max_val):
    """
    Return index of closest value in a to max_val
    """
    return np_abs(a - max_val).argmin()


def closest_less_than_index(a, max_val):
    """
    Return index of closest value in a to max_val which is less than max_val
    """
    less_than = a < max_val
    return closest_index(a[less_than], max_val)
