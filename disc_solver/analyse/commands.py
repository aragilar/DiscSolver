# -*- coding: utf-8 -*-
"""
Analysis commands
"""

from math import pi, sqrt

import numpy as np

import matplotlib as mpl
mpl.use("Qt4Agg")
mpl.rcParams["backend.qt4"] = "PySide"
import matplotlib.pyplot as plt

from . import generate_plot, get_plot_args
from ..utils import is_supersonic, find_in_array

INPUT_FORMAT = " {: <20}: {}"
INIT_FORMAT = " {: <20}: {}"
OTHER_FORMAT = " {: <20}: {}"


def info(inp, cons, angles, soln, args):
    """
    Output info about the solution
    """
    if args.get("input"):
        print("input settings:")
        for name, value in vars(inp).items():
            print(INPUT_FORMAT.format(name, value))
    if args.get("initial_conditions"):
        print("initial conditions:")
        for name, value in vars(cons).items():
            print(INIT_FORMAT.format(name, value))
    print("other info: ")
    if args.get("sound_ratio"):
        print(OTHER_FORMAT.format(
            "v_a/c_s at midplane",
            sqrt(inp.B_θ**2 / (4*pi*inp.ρ)) / inp.c_s
        ))
    if args.get("sonic_points"):
        zero_soln = np.zeros(len(soln))
        v = np.array([zero_soln, zero_soln, soln[:, 5]])
        slow_index = find_in_array(is_supersonic(
            v.T, soln[:, 0:3], soln[:, 6], inp.c_s, "slow"
        ), True)
        alfven_index = find_in_array(is_supersonic(
            v.T, soln[:, 0:3], soln[:, 6], inp.c_s, "alfven"
        ), True)
        fast_index = find_in_array(is_supersonic(
            v.T, soln[:, 0:3], soln[:, 6], inp.c_s, "fast"
        ), True)
        print(OTHER_FORMAT.format(
            "slow sonic point",
            180 * angles[slow_index] / pi if slow_index else None
        ))
        print(OTHER_FORMAT.format(
            "alfven sonic point",
            180 * angles[alfven_index] / pi if alfven_index else None
        ))
        print(OTHER_FORMAT.format(
            "fast sonic point",
            180 * angles[fast_index] / pi if fast_index else None
        ))


def plot(inp, cons, angles, soln, args):
    """
    Plot solution to file
    """
    plot_args = get_plot_args(args)
    # pylint: disable=star-args
    fig = generate_plot(angles, soln, inp, cons, **plot_args)
    fig.savefig(args["plot_filename"])


def show(inp, cons, angles, soln, args):
    """
    Show solution
    """
    plot_args = get_plot_args(args)
    # pylint: disable=star-args
    generate_plot(angles, soln, inp, cons, **plot_args)
    plt.show()
