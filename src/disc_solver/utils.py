# -*- coding: utf-8 -*-
"""
Useful functions
"""

from enum import IntEnum
from math import pi
from configparser import ConfigParser
from pathlib import Path

import numpy as np
from numpy import cos, sin, sqrt

import logbook

from stringtopy import (
    str_to_float_converter, str_to_int_converter, str_to_bool_converter
)

from .constants import G, AU, M_SUN

logger = logbook.Logger(__name__)

str_to_float = str_to_float_converter()
str_to_int = str_to_int_converter()
str_to_bool = str_to_bool_converter()


class MHD_Wave_Index(IntEnum):
    """
    Enum for MHD wave speed indexes
    """
    slow = 0
    alfven = 1
    fast = 2


def is_supersonic(v, B, rho, sound_speed, mhd_wave_type):
    """
    Checks whether velocity is supersonic.
    """
    speeds = mhd_wave_speeds(B, rho, sound_speed)

    v_axis = 1 if v.ndim > 1 else 0
    v_sq = np.sum(v**2, axis=v_axis)

    with np.errstate(invalid="ignore"):
        return v_sq > speeds[MHD_Wave_Index[mhd_wave_type]]


def mhd_wave_speeds(B, rho, sound_speed):
    """
    Computes MHD wave speeds (slow, alfven, fast)
    """
    B_axis = 1 if B.ndim == 2 else 0

    B_sq = np.sum(B**2, axis=B_axis)

    if B_axis:
        cos_sq_psi = B[:, ODEIndex.B_θ]**2 / B_sq
    else:
        cos_sq_psi = B[ODEIndex.B_θ]**2 / B_sq

    v_a_sq = B_sq / (4*pi*rho)
    slow = 1/2 * (
        v_a_sq + sound_speed**2 - sqrt(
            (v_a_sq + sound_speed**2)**2 -
            4 * v_a_sq * sound_speed**2 * cos_sq_psi
        )
    )
    alfven = v_a_sq * cos_sq_psi
    fast = 1/2 * (
        v_a_sq + sound_speed**2 + sqrt(
            (v_a_sq + sound_speed**2)**2 -
            4 * v_a_sq * sound_speed**2 * cos_sq_psi
        )
    )
    return slow, alfven, fast


def cot(angle):
    """
    Computes cot
    """
    if angle == pi:
        return float('-inf')
    elif angle == pi/2:
        return 0
    elif angle == 0:
        return float('inf')
    return cos(angle)/sin(angle)


def sec(angle):
    """
    Computes sec
    """
    return 1 / cos(angle)


def cosec(angle):
    """
    Computes cosec
    """
    return 1 / sin(angle)


def find_in_array(array, item):
    """
    Finds item in array or returns None
    """
    if array.ndim > 1:
        raise TypeError("array must be 1D")
    try:
        return list(array).index(item)
    except ValueError:
        return None


def get_normalisation(inp, radius=AU, mass=M_SUN, density=1.5e-9):
    """
    Get normalisation based on location and density
    """
    kepler_sq = G * mass / radius
    c_s = inp.c_s_on_v_k * sqrt(kepler_sq)
    v_a = inp.v_a_on_c_s * c_s
    return {
        "v_norm": c_s,
        "B_norm": v_a * 2 * sqrt(pi * density),
        "ρ_norm": density,
        "η_norm": radius * c_s,
    }


def expanded_path(*path):
    """
    Return expanded pathlib.Path object
    """
    return Path(*path).expanduser().absolute()  # pylint: disable=no-member


class CaseDependentConfigParser(ConfigParser):
    # pylint: disable=too-many-ancestors
    """
    configparser.ConfigParser subclass that removes the case transform.
    """
    def optionxform(self, optionstr):
        return optionstr


def get_solutions(run, soln_range):
    """
    Get solutions based on range
    """
    if soln_range is None:
        soln_range = "0"
    elif soln_range == "final":
        return run.final_solution
    return run.solutions[soln_range]


class ODEIndex(IntEnum):
    """
    Enum for array index for variables in the odes
    """
    B_r = 0
    B_φ = 1
    B_θ = 2
    v_r = 3
    v_φ = 4
    v_θ = 5
    ρ = 6
    B_φ_prime = 7
    E_r = 7
    η_O = 8
    η_A = 9
    η_H = 10


MAGNETIC_INDEXES = [ODEIndex.B_r, ODEIndex.B_φ, ODEIndex.B_θ]
VELOCITY_INDEXES = [ODEIndex.v_r, ODEIndex.v_φ, ODEIndex.v_θ]


class DiscSolverError(Exception):
    """
    Base error class for DiscSolver
    """
    pass
