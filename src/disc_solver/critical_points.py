# -*- coding: utf-8 -*-
"""
Functions relating to the sonic and MHD points
"""
from numpy import sqrt, any as np_any, pi

from .utils import (
    ODEIndex, MHD_Wave_Index, interp_for_value, first_closest_index,
)


def mhd_wave_speeds(*, values, sound_speed=1):
    """
    Computes MHD wave speeds (slow, alfven, fast)
    """
    B_r = values[..., ODEIndex.B_r]
    B_φ = values[..., ODEIndex.B_φ]
    B_θ = values[..., ODEIndex.B_θ]
    v_r = values[..., ODEIndex.v_r]
    v_φ = values[..., ODEIndex.v_φ]
    v_θ = values[..., ODEIndex.v_θ]
    ρ = values[..., ODEIndex.ρ]

    B_sq = B_r ** 2 + B_φ ** 2 + B_θ ** 2
    v_sq = v_r ** 2 + v_φ ** 2 + v_θ ** 2

    cos_sq_psi = ((v_r * B_r + v_θ * B_θ) ** 2) / (v_sq * B_sq)

    v_a_sq = B_sq / (4 * pi * ρ)
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


def get_mach_numbers(solution):
    """
    Compute the mach numbers of the solution. Order is slow, sonic, alfven,
    fast.
    """
    soln = solution.solution

    v_θ = soln[..., ODEIndex.v_θ]
    v_sq = v_θ ** 2

    wave_speeds_sq = mhd_wave_speeds(values=soln)

    slow_mach = sqrt(v_sq / wave_speeds_sq[MHD_Wave_Index.slow])
    alfven_mach = sqrt(v_sq / wave_speeds_sq[MHD_Wave_Index.alfven])
    fast_mach = sqrt(v_sq / wave_speeds_sq[MHD_Wave_Index.fast])
    sonic_mach = v_θ
    return slow_mach, sonic_mach, alfven_mach, fast_mach


def get_critical_point_indices(solution):
    """
    Get the index at which each of the critical points occur closest to. Order
    is slow, sonic, alfven, fast.
    """
    slow_mach, sonic_mach, alfven_mach, fast_mach = get_mach_numbers(solution)

    slow_index = first_closest_index(slow_mach, 1)
    sonic_index = first_closest_index(sonic_mach, 1)
    alfven_index = first_closest_index(alfven_mach, 1)
    fast_index = first_closest_index(fast_mach, 1)

    return slow_index, sonic_index, alfven_index, fast_index


def get_all_sonic_points(solution):
    """
    Get the angles at which the MHD sonic points occurs
    """
    slow_mach, sonic_mach, alfven_mach, fast_mach = get_mach_numbers(solution)
    angles = solution.angles
    if np_any(slow_mach > 1.0):
        slow_angle = interp_for_value(xs=angles, ys=slow_mach, y=1)
    else:
        slow_angle = None
    if np_any(sonic_mach > 1.0):
        sonic_angle = interp_for_value(xs=angles, ys=sonic_mach, y=1)
    else:
        sonic_angle = None
    if np_any(alfven_mach > 1.0):
        alfven_angle = interp_for_value(xs=angles, ys=alfven_mach, y=1)
    else:
        alfven_angle = None
    if np_any(fast_mach > 1.0):
        fast_angle = interp_for_value(xs=angles, ys=fast_mach, y=1)
    else:
        fast_angle = None
    return slow_angle, sonic_angle, alfven_angle, fast_angle
