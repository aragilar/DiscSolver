# -*- coding: utf-8 -*-
"""
Useful functions
"""

from math import pi, cos, sin, sqrt
from configparser import ConfigParser
from pathlib import Path

import numpy as np
from matplotlib.ticker import FuncFormatter

from stringtopy import (
    str_to_float_converter, str_to_int_converter, str_to_bool_converter
)

from .constants import G, AU, M_SUN

MHD_WAVE_INDEX = {"slow": 0, "alfven": 1, "fast": 2}

str_to_float = str_to_float_converter()
str_to_int = str_to_int_converter()
str_to_bool = str_to_bool_converter()


def strdict(d):
    """ stringify a dictionary"""
    return {str(k): str(v) for k, v in d.items()}


def is_supersonic(v, B, rho, sound_speed, mhd_wave_type):
    """
    Checks whether velocity is supersonic.
    """
    speeds = mhd_wave_speeds(v, B, rho, sound_speed)

    v_axis = 1 if v.ndim > 1 else 0
    v_sq = np.sum(v**2, axis=v_axis)
    return v_sq > speeds[MHD_WAVE_INDEX[mhd_wave_type]]


def mhd_wave_speeds(v, B, rho, sound_speed):
    """
    Computes MHD wave speeds (slow, alfven, fast)
    """
    v_axis = 1 if v.ndim > 1 else 0
    B_axis = 1 if B.ndim > 1 else 0

    v_sq = np.sum(v**2, axis=v_axis)
    B_sq = np.sum(B**2, axis=B_axis)

    cos_sq_psi = np.sum(
        (v.T**2/v_sq).T * (B.T**2/B_sq).T, axis=max(v_axis, B_axis)
    ) ** 2

    v_a_sq = B_sq / (4*pi*rho)
    slow = 1/2 * (
        v_a_sq + sound_speed**2 - np.sqrt(
            (v_a_sq + sound_speed**2)**2 -
            4 * v_a_sq * sound_speed**2 * cos_sq_psi
        )
    )
    alfven = v_a_sq * cos_sq_psi
    fast = 1/2 * (
        v_a_sq + sound_speed**2 + np.sqrt(
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


def better_sci_format(physical_axis):
    """
    Use scientific notation for each tick mark for axis `physical_axis`.
    """
    physical_axis.set_major_formatter(
        FuncFormatter(lambda x, pos: "{:.2e}".format(x).replace("-", "−"))
    )


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


def cli_to_var(cmd):
    """
    Convert cli style argument to valid python name
    """
    return cmd.replace("-", "_")


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


def allvars(obj):
    """
    vars replacement to work on namedtuples
    """
    try:
        return vars(obj)
    except TypeError as e:
        try:
            return obj._asdict()
        except AttributeError as f:
            raise e from f  # maybe?


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


try:
    from os import fspath  # pylint: disable=unused-import,wrong-import-order
except ImportError:
    def fspath(path):
        """Return the string representation of the path.

        If str or bytes is passed in, it is returned unchanged.
        """
        if isinstance(path, (str, bytes)):
            return path

        # Work from the object's type to match method resolution of other magic
        # methods.
        path_type = type(path)
        try:
            return path_type.__fspath__(path)
        except AttributeError:
            if hasattr(path_type, '__fspath__'):
                raise
            try:
                import pathlib
            except ImportError:
                pass
            else:
                if isinstance(path, pathlib.PurePath):
                    return str(path)

            raise TypeError(
                "expected str, bytes or os.PathLike object, not " +
                path_type.__name__
            )
