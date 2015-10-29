# -*- coding: utf-8 -*-
"""
Stuff to analyse solutions
"""

from math import sqrt

import numpy as np
from numpy import degrees
import matplotlib.pyplot as plt

from ..constants import KM
from ..utils import (
    better_sci_format, mhd_wave_speeds, MHD_WAVE_INDEX, get_normalisation
)


def generate_plot(soln_file, **kwargs):
    """
    Generate plot, with enough freedom to be able to format fig
    """
    soln_range = kwargs.pop("soln_range", "0")
    soln_instance = get_solutions(soln_file, soln_range)
    soln = soln_instance.solution
    angles = soln_instance.angles
    cons = soln_instance.initial_conditions
    inp = soln_instance.soln_input

    norms = get_normalisation(inp)  # need to allow config here
    B_norm, v_norm, ρ_norm = norms["B_norm"], norms["v_norm"], norms["ρ_norm"]
    zero_soln = np.zeros(len(soln))
    v = np.array([zero_soln, zero_soln, soln[:, 5]])
    wave_speeds = np.sqrt(mhd_wave_speeds(
        v.T, soln[:, 0:3], soln[:, 6], cons.c_s * v_norm
    ))
    linestyle = kwargs.pop("line style")
    with_slow = kwargs.pop("with slow")
    with_alfven = kwargs.pop("with alfven")
    with_fast = kwargs.pop("with fast")
    with_sonic = kwargs.pop("with sonic")

    param_names = [
        {
            "name": "B_r",
            "y_label": "Magnetic Field (Gauss)",
            "normalisation": B_norm,
        },
        {
            "name": "B_φ",
            "y_label": "Magnetic Field (Gauss)",
            "normalisation": B_norm,
        },
        {
            "name": "B_θ",
            "y_label": "Magnetic Field (Gauss)",
            "normalisation": B_norm,
        },
        {
            "name": "v_r",
            "y_label": "Velocity Field (km/s)",
            "normalisation": v_norm / KM,  # km/s
        },
        {
            "name": "v_φ",
            "y_label": "Velocity Field (km/s)",
            "normalisation": v_norm / KM,  # km/s
            "offset": sqrt(cons.norm_kepler_sq) * v_norm / KM
        },
        {
            "name": "v_θ",
            "y_label": "Velocity Field (km/s)",
            "normalisation": v_norm / KM,  # km/s
            "legend": True,
            "scale": kwargs.pop("v_θ scale", "linear"),
            "extras": []
        },
        {
            "name": "ρ",
            "y_label": "Density ($g cm^{-3}$)",
            "normalisation": ρ_norm,
            "scale": "log",
        },
        {
            "name": "B_φ_prime",
            "y_label": "Magnetic Field (Gauss)",
            "normalisation": B_norm,
        },
    ]

    if with_slow:
        param_names[5]["extras"].append({
            "label": "slow",
            "data": wave_speeds[MHD_WAVE_INDEX["slow"]],
            "normalisation": v_norm / KM,
        })
    if with_alfven:
        param_names[5]["extras"].append({
            "label": "alfven",
            "data": wave_speeds[MHD_WAVE_INDEX["alfven"]],
            "normalisation": v_norm / KM,
        })
    if with_fast:
        param_names[5]["extras"].append({
            "label": "fast",
            "data": wave_speeds[MHD_WAVE_INDEX["fast"]],
            "normalisation": v_norm / KM,
        })
    if with_sonic:
        param_names[5]["extras"].append({
            "label": "sound",
            "data": np.ones(len(soln)),
            "normalisation": v_norm / KM,
        })

    fig, axes = plt.subplots(
        nrows=2, ncols=4, tight_layout=True, sharex=True, **kwargs
    )
    axes.shape = len(param_names)
    for i, settings in enumerate(param_names):
        ax = axes[i]
        ax.plot(
            degrees(angles),
            (
                soln[:, i] * settings["normalisation"] -
                settings.get("offset", 0)
            ), linestyle,
        )
        for extra in settings.get("extras", []):
            ax.plot(
                degrees(angles),
                extra["data"] * extra["normalisation"],
                label=extra.get("label")
            )
        ax.set_xlabel("angle from plane (°)")
        ax.set_ylabel(settings["y_label"])
        ax.set_yscale(settings.get("scale", "linear"))
        ax.set_title(settings["name"])
        if settings.get("legend"):
            ax.legend()
        better_sci_format(ax.yaxis)
    fig.suptitle("{}:{}".format(
        soln_file.root.config_filename,
        soln_file.root.config_input.label
    ))
    return fig


def plot_options(parser):
    """
    Add cli arguments for defining plot
    """
    parser.add_argument("--v_θ", choices=("log", "linear"), default="linear")
    parser.add_argument("--line-style", default="-")
    parser.add_argument(
        "--with-slow", action='store_true', default=False)
    parser.add_argument(
        "--with-alfven", action='store_true', default=False)
    parser.add_argument(
        "--with-fast", action='store_true', default=False)
    parser.add_argument(
        "--with-sonic", action='store_true', default=False)


def get_plot_args(args):
    """
    Parse plot args
    """
    return {
        "v_θ scale": args.get("v_θ", "linear"),
        "with slow": args.get("with_slow", False),
        "with alfven": args.get("with_alfven", False),
        "with fast": args.get("with_fast", False),
        "with sonic": args.get("with_sonic", False),
        "line style": args.get("line_style", "-"),
        "soln_range": args.get("soln_range", "0"),
    }


def generate_deriv_plot(soln_file, **kwargs):
    """
    Generate plot of derivatives
    """
    param_names = [
        {
            "name": "B_r",
        },
        {
            "name": "B_φ",
        },
        {
            "name": "B_θ",
        },
        {
            "name": "v_r",
        },
        {
            "name": "v_φ",
        },
        {
            "name": "v_θ",
        },
        {
            "name": "ρ",
        },
        {
            "name": "B_φ_prime",
        },
    ]

    soln_range = kwargs.pop("soln_range", "0")
    linestyle = kwargs.pop("line style")
    soln_instance = get_solutions(soln_file, soln_range)
    internal_data = soln_instance.internal_data
    deriv_angles = internal_data.angles
    derivs = internal_data.derivs
    npnot = np.logical_not

    fig, axes = plt.subplots(
        nrows=2, ncols=4, tight_layout=True, sharex=True, **kwargs
    )
    axes.shape = len(param_names)
    for i, settings in enumerate(param_names):
        ax = axes[i]
        pos_deriv = derivs[:, i] >= 0
        ax.plot(
            degrees(deriv_angles[pos_deriv]),
            derivs[pos_deriv, i], linestyle + "b",
        )
        ax.plot(
            degrees(deriv_angles[npnot(pos_deriv)]),
            - derivs[npnot(pos_deriv), i], linestyle + "g",
        )
        ax.set_xlabel("angle from plane (°)")
        ax.set_yscale(settings.get("scale", "log"))
        ax.set_title(settings["name"])
        if settings.get("legend"):
            ax.legend()
        better_sci_format(ax.yaxis)
    return fig


def deriv_plot_options(parser):
    """
    Add cli arguments for defining derivative plot
    """
    parser.add_argument("--line-style", default=".")


def get_deriv_plot_args(args):
    """
    Parse plot args
    """
    return {
        "line style": args.get("line_style", "-"),
        "soln_range": args.get("soln_range", "0"),
    }


def generate_params_plot(soln_file, **kwargs):
    """
    Generate plot of all values, including intermediate values
    """
    param_names = [
        {
            "name": "B_r",
        },
        {
            "name": "B_φ",
        },
        {
            "name": "B_θ",
        },
        {
            "name": "v_r",
        },
        {
            "name": "v_φ",
        },
        {
            "name": "v_θ",
        },
        {
            "name": "ρ",
        },
        {
            "name": "B_φ_prime",
        },
    ]

    linestyle = kwargs.pop("line style")
    soln_range = kwargs.pop("soln_range", "0")
    soln_instance = get_solutions(soln_file, soln_range)
    internal_data = soln_instance.internal_data
    param_angles = internal_data.angles
    params = internal_data.params
    npnot = np.logical_not

    fig, axes = plt.subplots(
        nrows=2, ncols=4, tight_layout=True, sharex=True, **kwargs
    )
    axes.shape = len(param_names)
    for i, settings in enumerate(param_names):
        ax = axes[i]
        pos_params = params[:, i] >= 0
        ax.plot(
            degrees(param_angles[pos_params]),
            params[pos_params, i], linestyle,
        )
        ax.plot(
            degrees(param_angles[npnot(pos_params)]),
            - params[npnot(pos_params), i], linestyle,
        )
        ax.set_xlabel("angle from plane (°)")
        ax.set_yscale(settings.get("scale", "log"))
        ax.set_title(settings["name"])
        if settings.get("legend"):
            ax.legend()
        better_sci_format(ax.yaxis)
    return fig


def params_plot_options(parser):
    """
    Add cli arguments for defining derivative plot
    """
    parser.add_argument("--line-style", default=".")


def get_params_plot_args(args):
    """
    Parse plot args
    """
    return {
        "line style": args.get("line_style", "-"),
        "soln_range": args.get("soln_range", "0"),
    }


def get_solutions(soln_file, soln_range):
    """
    Get solutions based on range
    """
    if soln_range == "final":
        return soln_file.root.final_solution
    return soln_file.root.solutions[soln_range]
