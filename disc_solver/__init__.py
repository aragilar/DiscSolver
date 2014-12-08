# -*- coding: utf-8 -*-
"""
Solver for PHD Project
"""

__version__ = "0.1"
from math import pi

import logbook

import matplotlib as mpl
mpl.use("Qt4Agg")
mpl.rcParams["backend.qt4"] = "PySide"
import matplotlib.pyplot as plt

from .constants import AU, M_SUN, KM
from .analyse import generate_plot
from .solution import solution
from .config import define_conditions

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

    v_rin_on_v_k = 0.1

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

    (
        angles, init_con, c_s, norm_kepler_sq, η_O, η_A, η_H, v_norm, B_norm,
        ρ_norm
    ) = define_conditions(
        central_mass, radius, v_rin_on_v_k, B_θ, ρ, η_O, η_A, η_H, c_s, β,
        start, stop
    )

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
