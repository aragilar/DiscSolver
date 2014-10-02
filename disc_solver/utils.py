# -*- coding: utf-8 -*-
"""
"""

from math import pi, cos, sin

import numpy as np


def is_supersonic(v, B, rho, sound_speed, mhd_wave_type):
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
    if angle == pi:
        return float('-inf')
    elif angle == pi/2:
        return 0
    elif angle == 0:
        return float('inf')
    return cos(angle)/sin(angle)