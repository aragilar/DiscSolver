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

from .analyse import generate_plot
from .solution import solution
from .config import define_conditions, get_input

log = logbook.Logger(__name__)


def main():
    """
    The main function
    """
    inp = get_input()
    cons = define_conditions(inp)

    try:
        soln = solution(
            cons.angles, cons.init_con, inp.β, cons.c_s, cons.norm_kepler_sq,
            cons.η_O, cons.η_A, cons.η_H, max_steps=inp.max_steps,
            taylor_stop_angle=inp.taylor_stop_angle / 180 * pi
        )
    except RuntimeError as e:
        # pylint: disable=no-member
        angles = e.x_vals
        soln = e.y_vals

    fig = generate_plot(angles, soln, cons.B_norm, cons.v_norm, cons.ρ_norm)
    plt.show()
    fig.savefig("plot.png")
