# -*- coding: utf-8 -*-
"""
Analysis commands
"""

import numpy as np
from numpy import degrees

import matplotlib as mpl
mpl.use("Qt4Agg")
mpl.rcParams["backend.qt4"] = "PySide"
import matplotlib.pyplot as plt

from .plot_functions import (
    generate_plot, get_plot_args, generate_deriv_plot, get_deriv_plot_args,
    generate_params_plot, get_params_plot_args, get_solutions
)
from ..utils import is_supersonic, find_in_array, get_normalisation

INPUT_FORMAT = " {: <20}: {}"
INIT_FORMAT = " {: <20}: {}"
OTHER_FORMAT = " {: <20}: {}"


def info(soln_file, args):
    """
    Output info about the solution
    """
    print("run properties:")
    print("label: {}".format(soln_file.root.config_input.label))
    print("config filename: {}".format(soln_file.root.config_filename))
    print("file version: {}".format(soln_file.file.attrs["version"]))
    print("generator version: {}".format(
        soln_file.file.attrs["generator_version"]
    ))
    print("number of solutions: {}".format(len(soln_file.root.solutions)))

    soln_range = args.get("soln_range", "0")

    print("ODE return flag: {!s}".format(
        soln_file.root.solutions[soln_range].flag
    ))
    print("Coordinate System: {!s}".format(
        soln_file.root.solutions[soln_range].coordinate_system
    ))

    inp = soln_file.root.solutions[soln_range].soln_input
    init_con = soln_file.root.solutions[soln_range].initial_conditions
    v_norm = get_normalisation(inp)["v_norm"]  # need to fix config here
    c_s = init_con.c_s * v_norm
    if args.get("input"):
        print("input settings:")
        for name, value in vars(inp).items():
            print(INPUT_FORMAT.format(name, value))
    if args.get("initial_conditions"):
        print("initial conditions:")
        for name, value in vars(init_con).items():
            print(INIT_FORMAT.format(name, value))
    print("other info: ")
    if args.get("sonic_points"):
        soln = soln_file.root.solutions[soln_range].solution
        angles = soln_file.root.solutions[soln_range].angles
        zero_soln = np.zeros(len(soln))
        v = np.array([zero_soln, zero_soln, soln[:, 5]])
        slow_index = find_in_array(is_supersonic(
            v.T, soln[:, 0:3], soln[:, 6], c_s, "slow"
        ), True)
        alfven_index = find_in_array(is_supersonic(
            v.T, soln[:, 0:3], soln[:, 6], c_s, "alfven"
        ), True)
        fast_index = find_in_array(is_supersonic(
            v.T, soln[:, 0:3], soln[:, 6], c_s, "fast"
        ), True)
        print(OTHER_FORMAT.format(
            "slow sonic point",
            degrees(angles[slow_index]) if slow_index else None
        ))
        print(OTHER_FORMAT.format(
            "alfven sonic point",
            degrees(angles[alfven_index]) if alfven_index else None
        ))
        print(OTHER_FORMAT.format(
            "fast sonic point",
            degrees(angles[fast_index]) if fast_index else None
        ))


def plot(soln_file, args):
    """
    Plot solution to file
    """
    plot_args = get_plot_args(args)
    fig = generate_plot(soln_file, **plot_args)
    fig.savefig(args["plot_filename"])


def show(soln_file, args):
    """
    Show solution
    """
    plot_args = get_plot_args(args)
    generate_plot(soln_file, **plot_args)
    plt.show()


def deriv_show(soln_file, args):
    """
    Show derivatives
    """
    plot_args = get_deriv_plot_args(args)
    generate_deriv_plot(soln_file, **plot_args)
    plt.show()


def check_taylor(soln_file, args):
    """
    Compare derivatives from taylor series to full version
    """
    soln_range = args.get("soln_range", "0")
    soln_instance = get_solutions(soln_file, soln_range)
    v_r_normal = soln_instance.internal_data.v_r_normal
    v_φ_normal = soln_instance.internal_data.v_φ_normal
    ρ_normal = soln_instance.internal_data.ρ_normal
    v_r_taylor = soln_instance.internal_data.v_r_taylor
    v_φ_taylor = soln_instance.internal_data.v_φ_taylor
    ρ_taylor = soln_instance.internal_data.ρ_taylor

    deriv_angles = soln_instance.internal_data.angles
    # pylint: disable=unused-variable
    fig, axes = plt.subplots(ncols=3, tight_layout=True)
    if args.get("show_values", False):
        axes[0].plot(degrees(deriv_angles), v_r_normal)
        axes[0].plot(degrees(deriv_angles), v_r_taylor)
        axes[1].plot(degrees(deriv_angles), v_φ_normal)
        axes[1].plot(degrees(deriv_angles), v_φ_taylor)
        axes[2].plot(degrees(deriv_angles), ρ_normal)
        axes[2].plot(degrees(deriv_angles), ρ_taylor)
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
        axes[2].set_yscale("log")
    else:
        axes[0].plot(
            degrees(deriv_angles),
            np.abs(v_r_normal - v_r_taylor), '.'
        )
        axes[1].plot(
            degrees(deriv_angles),
            np.abs(v_φ_normal - v_φ_taylor), '.'
        )
        axes[2].plot(
            degrees(deriv_angles),
            np.abs(ρ_normal - ρ_taylor), '.'
        )
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
        axes[2].set_yscale("log")
    plt.show()


def params_show(soln_file, args):
    """
    Show solution at every step the solver takes.
    """
    plot_args = get_params_plot_args(args)
    generate_params_plot(soln_file, **plot_args)
    plt.show()
