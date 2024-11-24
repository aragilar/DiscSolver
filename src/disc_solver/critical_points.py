# -*- coding: utf-8 -*-
"""
Functions relating to the sonic and MHD points
"""
from numpy import (
    sqrt, any as np_any, argmin, abs as npabs, pi
)
from scipy.interpolate import interp1d

from .utils import (
    ODEIndex, MHD_Wave_Index,
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

    slow_index = argmin(npabs(1 - slow_mach))
    sonic_index = argmin(npabs(1 - sonic_mach))
    alfven_index = argmin(npabs(1 - alfven_mach))
    fast_index = argmin(npabs(1 - fast_mach))

    return slow_index, sonic_index, alfven_index, fast_index


def get_sonic_point(solution):
    """
    Get the angle at which the sonic point occurs
    """
    # This bit needs fixing, should check if sound speed is the root
    # if solution.t_roots is not None:
    #     return solution.t_roots[0]
    # Also should work out purpose of function - extrapolate or not, roots or
    # not. Currently only used in info
    fit = interp1d(
        solution.solution[:, ODEIndex.v_θ],
        solution.angles,
        fill_value="extrapolate",
    )
    return fit(1.0)


def get_all_sonic_points(solution):
    """
    Get the angles at which the mhd sonic points occurs
    """
    slow_mach, sonic_mach, alfven_mach, fast_mach = get_mach_numbers(solution)
    angles = solution.angles
    if np_any(slow_mach > 1.0):
        slow_angle = interp1d(slow_mach, angles)(1.0)
    else:
        slow_angle = None
    if np_any(sonic_mach > 1.0):
        sonic_angle = interp1d(sonic_mach, angles)(1.0)
    else:
        sonic_angle = None
    if np_any(alfven_mach > 1.0):
        alfven_angle = interp1d(alfven_mach, angles)(1.0)
    else:
        alfven_angle = None
    if np_any(fast_mach > 1.0):
        fast_angle = interp1d(fast_mach, angles)(1.0)
    else:
        fast_angle = None
    return slow_angle, sonic_angle, alfven_angle, fast_angle
