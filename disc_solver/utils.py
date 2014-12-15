# -*- coding: utf-8 -*-
"""
Useful functions
"""

from math import pi, cos, sin

import numpy as np
from matplotlib.ticker import FuncFormatter


def is_supersonic(v, B, rho, sound_speed, mhd_wave_type):
    """
    Checks whether velocity is supersonic.
    """
    v_axis = 1 if v.ndim > 1 else 0
    B_axis = 1 if B.ndim > 1 else 0

    v_sq = np.sum(v**2, axis=v_axis)
    B_sq = np.sum(B**2, axis=B_axis)

    cos_sq_psi = np.sum(
        (v.T**2/v_sq).T * (B.T**2/B_sq).T, axis=max(v_axis, B_axis)
    ) ** 2

    v_a_sq = B_sq / (4*pi*rho)
    if mhd_wave_type == "slow":
        mhd_wave_speed = 1/2 * (
            v_a_sq + sound_speed**2 - np.sqrt(
                (v_a_sq + sound_speed**2)**2 -
                4 * v_a_sq * sound_speed**2 * cos_sq_psi
            )
        )
    elif mhd_wave_type == "alfven":
        mhd_wave_speed = v_a_sq * cos_sq_psi
    elif mhd_wave_type == "fast":
        mhd_wave_speed = 1/2 * (
            v_a_sq + sound_speed**2 + np.sqrt(
                (v_a_sq + sound_speed**2)**2 -
                4 * v_a_sq * sound_speed**2 * cos_sq_psi
            )
        )

    return v_sq > mhd_wave_speed


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


def better_sci_format(physical_axis):
    """
    Use scientific notation for each tick mark for axis `physical_axis`.
    """
    physical_axis.set_major_formatter(
        FuncFormatter(lambda x, pos: "{:.2e}".format(x).replace("-", "−"))
    )
