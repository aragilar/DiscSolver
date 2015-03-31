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

from . import (
    generate_plot, get_plot_args, generate_deriv_plot, get_deriv_plot_args,
    generate_params_plot, get_params_plot_args,
)
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
    fig = generate_plot(angles, soln, inp, cons, **plot_args)
    fig.savefig(args["plot_filename"])


def show(inp, cons, angles, soln, args):
    """
    Show solution
    """
    plot_args = get_plot_args(args)
    generate_plot(angles, soln, inp, cons, **plot_args)
    plt.show()


def deriv_show(inp, cons, angles, soln, args):
    """
    Show derivatives
    """
    plot_args = get_deriv_plot_args(args)
    generate_deriv_plot(angles, soln, inp, cons, **plot_args)
    plt.show()


def check_taylor(inp, cons, angles, soln, args):
    """
    Compare derivatives from taylor series to full version
    """
    # pylint: disable=unused-argument
    internal_data = args["internal_data"]
    v_r_normal = np.array(internal_data["v_r normal"])
    v_φ_normal = np.array(internal_data["v_φ normal"])
    ρ_normal = np.array(internal_data["ρ normal"])
    v_r_taylor = np.array(internal_data["v_r taylor"])
    v_φ_taylor = np.array(internal_data["v_φ taylor"])
    ρ_taylor = np.array(internal_data["ρ taylor"])

    deriv_angles = np.array(internal_data["angles"])
    # pylint: disable=unused-variable
    fig, axes = plt.subplots(ncols=3, tight_layout=True)
    if args["show_values"]:
        axes[0].plot(90 - deriv_angles * 180 / pi, v_r_normal)
        axes[0].plot(90 - deriv_angles * 180 / pi, v_r_taylor)
        axes[1].plot(90 - deriv_angles * 180 / pi, v_φ_normal)
        axes[1].plot(90 - deriv_angles * 180 / pi, v_φ_taylor)
        axes[2].plot(90 - deriv_angles * 180 / pi, ρ_normal)
        axes[2].plot(90 - deriv_angles * 180 / pi, ρ_taylor)
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
        axes[2].set_yscale("log")
    else:
        axes[0].plot(
            90 - deriv_angles * 180 / pi,
            np.abs(v_r_normal - v_r_taylor), '.'
        )
        axes[1].plot(
            90 - deriv_angles * 180 / pi,
            np.abs(v_φ_normal - v_φ_taylor), '.'
        )
        axes[2].plot(
            90 - deriv_angles * 180 / pi,
            np.abs(ρ_normal - ρ_taylor), '.'
        )
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
        axes[2].set_yscale("log")
    plt.show()


def params_show(inp, cons, angles, soln, args):
    """
    Show solution at every step the solver takes.
    """
    plot_args = get_params_plot_args(args)
    generate_params_plot(angles, soln, inp, cons, **plot_args)
    plt.show()
