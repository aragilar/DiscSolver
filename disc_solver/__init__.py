# -*- coding: utf-8 -*-
"""
Solver for PHD Project
"""

__version__ = "0.1"
from math import pi, sqrt

import logbook

import numpy as np
import matplotlib as mpl
mpl.use("Qt4Agg")
mpl.rcParams["backend.qt4"] = "PySide"
import matplotlib.pyplot as plt

from .constants import G, AU, M_SUN, KM
from .analyse import generate_plot
from .solution import solution

log = logbook.Logger(__name__)


def main():
    """
    The main function
    """

    start = 90
    stop = 85
    taylor_stop_angle = 89.99

    # pick a radii, 1AU makes it easy to calculate
    radius = 1 * AU

    central_mass = 1 * M_SUN

    β = 4/3

    # B_θ is the equipartition field
    B_θ = 18  # G

    # from wardle 2007 for 1 AU
    # assume B ~ 1 G
    η_O = 5e15  # cm^2/s
    η_H = 5e16  # cm^2/s
    η_A = 1e14  # cm^2/s

    c_s = 0.99 * KM  # actually in cm/s for conversions
    # ρ computed from Wardle 2007
    ρ = 1.5e-9  # g/cm^3

    keplerian_velocity = sqrt(G * central_mass / radius)  # cm/s

    v_r = - 0.1 * keplerian_velocity

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

    if v_r > 0:
        log.error("v_r > 0")
        exit(1)

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

    try:
        soln = solution(
            angles, init_con, β, c_s, norm_kepler_sq, η_O,
            η_A, η_H, max_steps=10000,
            taylor_stop_angle=taylor_stop_angle / 180 * pi
        )
    except RuntimeError as e:
        # pylint: disable=no-member
        angles = e.x_vals
        soln = e.y_vals

    fig = generate_plot(angles, soln, B_norm, v_norm, ρ_norm)
    plt.show()
    fig.savefig("plot.png")
