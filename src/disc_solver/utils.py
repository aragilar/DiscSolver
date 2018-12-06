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
DIFFUSIVE_INDEXES = [ODEIndex.η_O, ODEIndex.η_A, ODEIndex.η_H]


class CylindricalODEIndex(IntEnum):
    """
    Enum for array index for variables in the odes
    """
    B_R = 0
    B_φ = 1
    B_z = 2
    v_R = 3
    v_φ = 4
    v_z = 5
    ρ = 6
    B_φ_prime = 7
    E_R = 7
    η_O = 8
    η_A = 9
    η_H = 10


class DiscSolverError(Exception):
    """
    Base error class for DiscSolver
    """
    pass


def scale_solution_to_radii(soln, new_r, *, γ, use_E_r):
    """
    Scale spherical solution to given radii
    """
    v_scale = 1 / sqrt(new_r)
    B_scale = new_r ** (γ - 5 / 4)
    ρ_scale = new_r ** (2 * γ - 3 / 2)
    η_scale = sqrt(new_r)

    scaled_soln = np.full(soln.shape, np.nan, dtype=soln.dtype)

    if scaled_soln.ndim == 1:
        scaled_soln[VELOCITY_INDEXES] = v_scale * soln[VELOCITY_INDEXES]
        scaled_soln[MAGNETIC_INDEXES] = B_scale * soln[MAGNETIC_INDEXES]
        scaled_soln[ODEIndex.ρ] = ρ_scale * soln[ODEIndex.ρ]
        scaled_soln[DIFFUSIVE_INDEXES] = η_scale * soln[DIFFUSIVE_INDEXES]
        if use_E_r:
            raise DiscSolverError("E_r scaling not added yet")
        else:
            scaled_soln[ODEIndex.B_φ_prime] = B_scale * soln[
                ODEIndex.B_φ_prime
            ]
    else:
        scaled_soln[:, VELOCITY_INDEXES] = (v_scale * soln[
            :, VELOCITY_INDEXES
        ].T).T
        scaled_soln[:, MAGNETIC_INDEXES] = (B_scale * soln[
            :, MAGNETIC_INDEXES
        ].T).T
        scaled_soln[:, ODEIndex.ρ] = ρ_scale * soln[:, ODEIndex.ρ]
        scaled_soln[:, DIFFUSIVE_INDEXES] = (η_scale * soln[
            :, DIFFUSIVE_INDEXES
        ].T).T
        if use_E_r:
            raise DiscSolverError("E_r scaling not added yet")
        else:
            scaled_soln[:, ODEIndex.B_φ_prime] = B_scale * soln[
                :, ODEIndex.B_φ_prime
            ]

    return scaled_soln


def convert_solution_to_vertical(angles, soln, *, γ, c_s_on_v_k, use_E_r):
    """
    Shift solution values to vertical. Does not change to cylindrical
    """
    heights = sqrt((1 - cos(angles)) / cos(angles)) / c_s_on_v_k
    scaling = 1 + (1 - cos(angles)) / (c_s_on_v_k ** 2 * cos(angles))

    new_soln = scale_solution_to_radii(soln, scaling, γ=γ, use_E_r=use_E_r)

    return heights, new_soln


def spherical_r_θ_to_cylindrical_R_z(r_var, θ_var, angles):
    """
    Move r, θ in (modified) spherical coordinates to R, z in cylindrical
    coordinates
    """
    R_var = r_var * cos(angles) - θ_var * sin(angles)
    z_var = θ_var * cos(angles) + r_var * sin(angles)
    return R_var, z_var


def convert_spherical_vertical_to_cylindrical(angles, soln, *, use_E_r):
    """
    Move spherical variables at constant cylindrical radius to cylindrical
    variables
    """
    if use_E_r:
        raise DiscSolverError("E_r conversion not added yet")

    new_soln = np.full(soln.shape, np.nan, dtype=soln.dtype)

    if new_soln.ndim == 1:
        new_soln[CylindricalODEIndex.v_φ] = soln[ODEIndex.v_φ]
        new_soln[CylindricalODEIndex.B_φ] = soln[ODEIndex.B_φ]
        new_soln[CylindricalODEIndex.ρ] = soln[ODEIndex.ρ]
        new_soln[CylindricalODEIndex.η_O] = soln[ODEIndex.η_O]
        new_soln[CylindricalODEIndex.η_A] = soln[ODEIndex.η_A]
        new_soln[CylindricalODEIndex.η_H] = soln[ODEIndex.η_H]

        v_R, v_z = spherical_r_θ_to_cylindrical_R_z(
            soln[ODEIndex.v_r], soln[ODEIndex.v_θ], angles
        )
        new_soln[CylindricalODEIndex.v_R] = v_R
        new_soln[CylindricalODEIndex.v_z] = v_z

        B_R, B_z = spherical_r_θ_to_cylindrical_R_z(
            soln[ODEIndex.B_r], soln[ODEIndex.B_θ], angles
        )
        new_soln[CylindricalODEIndex.B_R] = B_R
        new_soln[CylindricalODEIndex.B_z] = B_z

        if use_E_r:
            raise DiscSolverError("E_r conversion not added yet")
        else:
            new_soln[CylindricalODEIndex.B_φ_prime] = soln[ODEIndex.B_φ_prime]

    else:
        new_soln[:, CylindricalODEIndex.v_φ] = soln[:, ODEIndex.v_φ]
        new_soln[:, CylindricalODEIndex.B_φ] = soln[:, ODEIndex.B_φ]
        new_soln[:, CylindricalODEIndex.ρ] = soln[:, ODEIndex.ρ]
        new_soln[:, CylindricalODEIndex.η_O] = soln[:, ODEIndex.η_O]
        new_soln[:, CylindricalODEIndex.η_A] = soln[:, ODEIndex.η_A]
        new_soln[:, CylindricalODEIndex.η_H] = soln[:, ODEIndex.η_H]

        v_R, v_z = spherical_r_θ_to_cylindrical_R_z(
            soln[:, ODEIndex.v_r], soln[:, ODEIndex.v_θ], angles
        )
        new_soln[:, CylindricalODEIndex.v_R] = v_R
        new_soln[:, CylindricalODEIndex.v_z] = v_z

        B_R, B_z = spherical_r_θ_to_cylindrical_R_z(
            soln[:, ODEIndex.B_r], soln[:, ODEIndex.B_θ], angles
        )
        new_soln[:, CylindricalODEIndex.B_R] = B_R
        new_soln[:, CylindricalODEIndex.B_z] = B_z

        if use_E_r:
            raise DiscSolverError("E_r conversion not added yet")
        else:
            new_soln[:, CylindricalODEIndex.B_φ_prime] = soln[
                :, ODEIndex.B_φ_prime
            ]

    return new_soln
