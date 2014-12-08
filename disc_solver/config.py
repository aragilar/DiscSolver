# -*- coding: utf-8 -*-
"""
Define input and environment for ode system
"""

from math import pi, sqrt

import logbook

import numpy as np

from .constants import G

log = logbook.Logger(__name__)


def define_conditions(
    central_mass, radius, v_rin_on_v_k, B_θ, ρ, η_O, η_A, η_H, c_s, β, start,
    stop
):
    """
    Compute initial conditions based on input
    """
    keplerian_velocity = sqrt(G * central_mass / radius)  # cm/s

    v_r = - v_rin_on_v_k * keplerian_velocity
    if v_r > 0:
        log.error("v_r > 0")
        exit(1)

    v_θ = 0  # symmetry across disc
    B_r = 0  # symmetry across disc
    B_φ = 0  # symmetry across disc

    # solution for A * v_φ**2 + B * v_φ + C = 0
    A_v_φ = 1
    B_v_φ = - (v_r * η_H) / (2 * (η_O + η_A))
    C_v_φ = (
        v_r**2 / 2 + 2 * β * c_s**2 -
        keplerian_velocity**2 +
        B_θ**2 * (
            2 * (β - 1) - v_r / (η_O + η_A)
        ) / (4 * pi * ρ)
    )
    v_φ = - 1/2 * (B_v_φ - sqrt(B_v_φ**2 - 4 * A_v_φ * C_v_φ))

    B_φ_prime = (v_φ * v_r * 4 * pi * ρ) / (2 * B_θ)

    log.debug("A_v_φ: {}".format(A_v_φ))
    log.debug("B_v_φ: {}".format(B_v_φ))
    log.debug("C_v_φ: {}".format(C_v_φ))
    log.info("v_φ: {}".format(v_r))
    log.info("B_φ_prime: {}".format(B_φ_prime))

    v_norm = c_s
    B_norm = B_θ
    diff_norm = v_norm * radius

    ρ_norm = B_norm**2 / (4 * pi * v_norm**2)

    init_con = np.zeros(8)

    init_con[0] = B_r / B_norm
    init_con[1] = B_φ / B_norm
    init_con[2] = B_θ / B_norm
    init_con[3] = v_r / v_norm
    init_con[4] = v_φ / v_norm
    init_con[5] = v_θ / v_norm
    init_con[6] = ρ / ρ_norm
    init_con[7] = B_φ_prime / B_norm

    norm_kepler_sq = keplerian_velocity**2 / v_norm**2
    c_s = c_s / v_norm
    η_O = η_O / diff_norm
    η_A = η_A / diff_norm
    η_H = η_H / diff_norm

    angles = np.linspace(start, stop, 10000) / 180 * pi
    return (
        angles, init_con, c_s, norm_kepler_sq, η_O, η_A, η_H, v_norm, B_norm,
        ρ_norm
    )
