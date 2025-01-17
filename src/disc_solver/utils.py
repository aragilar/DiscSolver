# -*- coding: utf-8 -*-
"""
Useful functions
"""

from argparse import ArgumentParser
from configparser import ConfigParser
from contextlib import contextmanager
from enum import IntEnum
from functools import wraps
from io import IOBase
from math import pi
from multiprocessing import Pool
from pathlib import Path
from sys import stdout, argv as sys_argv

import numpy as np
from numpy import cos, sin, sqrt, tan, diff, all as np_all

import logbook

from stringtopy import (
    str_to_float_converter, str_to_int_converter, str_to_bool_converter
)

from . import __version__ as ds_version
from .constants import G, AU, M_SUN
from .logging import logging_options

logger = logbook.Logger(__name__)

str_to_float = str_to_float_converter(use_none_on_fail=True)
str_to_int = str_to_int_converter(use_none_on_fail=True)
str_to_bool = str_to_bool_converter()


class DiscSolverError(Exception):
    """
    Base error class for DiscSolver
    """
    pass


class MHD_Wave_Index(IntEnum):
    """
    Enum for MHD wave speed indexes
    """
    slow = 0
    alfven = 1
    fast = 2


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


class CylindricalODEIndex(IntEnum):
    """
    Enum for array index for variables in the odes
    """
    B_r = 0
    B_φ = 1
    B_z = 2
    v_r = 3
    v_φ = 4
    v_z = 5
    ρ = 6
    B_φ_prime = 7
    E_r = 7
    η_O = 8
    η_A = 9
    η_H = 10


MAGNETIC_INDEXES = [ODEIndex.B_r, ODEIndex.B_φ, ODEIndex.B_θ]
VELOCITY_INDEXES = [ODEIndex.v_r, ODEIndex.v_φ, ODEIndex.v_θ]
DIFFUSIVE_INDEXES = [ODEIndex.η_O, ODEIndex.η_A, ODEIndex.η_H]
VERT_MAGNETIC_INDEXES = [
    CylindricalODEIndex.B_r, CylindricalODEIndex.B_φ, CylindricalODEIndex.B_z
]
VERT_VELOCITY_INDEXES = [
    CylindricalODEIndex.v_r, CylindricalODEIndex.v_φ, CylindricalODEIndex.v_z
]
VERT_DIFFUSIVE_INDEXES = [
    CylindricalODEIndex.η_O, CylindricalODEIndex.η_A, CylindricalODEIndex.η_H
]


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


def get_heights(angles, *, c_s_on_v_k):
    """
    Get heights from angles
    """
    return tan(angles) / c_s_on_v_k


def get_vertical_scaling(angles, *, c_s_on_v_k):
    """
    Get scaling from spherical to vertical_scale
    """
    heights = get_heights(angles, c_s_on_v_k=c_s_on_v_k)
    return sqrt(1 + heights**2)


def convert_solution_to_vertical(angles, soln, *, γ, c_s_on_v_k, use_E_r):
    """
    Shift solution values to vertical. Does not change to cylindrical
    """
    heights = get_heights(angles, c_s_on_v_k=c_s_on_v_k)
    scaling = get_vertical_scaling(angles, c_s_on_v_k=c_s_on_v_k)

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


def convert_spherical_to_cylindrical(angles, soln, *, use_E_r, γ, c_s_on_v_k):
    """
    Move spherical variables to cylindrical variables at constant cylindrical
    radius
    """
    if use_E_r:
        raise DiscSolverError("E_r conversion not added yet")

    heights, soln = convert_solution_to_vertical(
        angles, soln, use_E_r=use_E_r, γ=γ, c_s_on_v_k=c_s_on_v_k,
    )

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
        new_soln[CylindricalODEIndex.v_r] = v_R
        new_soln[CylindricalODEIndex.v_z] = v_z

        B_R, B_z = spherical_r_θ_to_cylindrical_R_z(
            soln[ODEIndex.B_r], soln[ODEIndex.B_θ], angles
        )
        new_soln[CylindricalODEIndex.B_r] = B_R
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
        new_soln[:, CylindricalODEIndex.v_r] = v_R
        new_soln[:, CylindricalODEIndex.v_z] = v_z

        B_R, B_z = spherical_r_θ_to_cylindrical_R_z(
            soln[:, ODEIndex.B_r], soln[:, ODEIndex.B_θ], angles
        )
        new_soln[:, CylindricalODEIndex.B_r] = B_R
        new_soln[:, CylindricalODEIndex.B_z] = B_z

        if use_E_r:
            raise DiscSolverError("E_r conversion not added yet")
        else:
            new_soln[:, CylindricalODEIndex.B_φ_prime] = soln[
                :, ODEIndex.B_φ_prime
            ]

    return heights, new_soln


@contextmanager
def open_or_stream(filename, *args, **kwargs):
    """
    Open a file or use existing stream.

    Passing '-' as the filename uses stdout.
    """
    if filename == '-':
        file = None
        yield stdout
    elif isinstance(filename, IOBase):
        file = None
        yield filename
    else:
        # pylint: disable=consider-using-with
        file = open(filename, *args, **kwargs)
        yield file

    if file is not None:
        file.close()


def main_entry_point_wrapper(description, **kwargs):
    """
    Wrapper for all entry points
    """
    def decorator(cmd):
        """
        decorator for entry points
        """
        @wraps(cmd)
        def wrap_main(argv=None):
            """
            Actual main function for analysis code, deals with parsers and
            logging
            """
            if argv is None:
                argv = sys_argv[1:]
            parser = ArgumentParser(
                description=description,
                **kwargs
            )
            parser.add_argument(
                '--version', action='version', version='%(prog)s ' + ds_version
            )
            logging_options(parser)
            return cmd(argv=argv, parser=parser)

        return wrap_main

    return decorator


@contextmanager
def nicer_mp_pool(*args, **kwargs):
    """
    Wrapper around multiprocessing.Pool - maybe look at forks or alternatives?
    """
    try:
        pool = Pool(*args, **kwargs)  # pylint: disable=consider-using-with
        yield pool
    finally:
        # terminate is usually called, which can break stuff if there's a
        # problem
        pool.close()
        pool.join()


def is_monotonically_increasing(arr):
    """
    Return if array is monotonically increasing.
    """
    return np_all(diff(arr) > 0)
