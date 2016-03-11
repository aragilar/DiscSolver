# -*- coding: utf-8 -*-
"""
Stuff to analyse solutions
"""

from collections import OrderedDict
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
    inp = soln_instance.solution_input
    y_roots = soln_instance.y_roots
    t_roots = soln_instance.t_roots

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
    stop = float(kwargs.pop("stop", 90))

    indexes = degrees(angles) <= stop

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
            degrees(angles[indexes]),
            (
                soln[:, i] * settings["normalisation"] -
                settings.get("offset", 0)
            )[indexes], linestyle,
        )
        for extra in settings.get("extras", []):
            ax.plot(
                degrees(angles[indexes]),
                (extra["data"] * extra["normalisation"])[indexes],
                label=extra.get("label")
            )
        ax.set_xlabel("angle from plane (°)")
        ax.set_ylabel(settings["y_label"])
        ax.set_yscale(settings.get("scale", "linear"))
        ax.set_title(settings["name"])
        if settings.get("legend"):
            ax.legend()
        better_sci_format(ax.yaxis)
        ax.axvline(degrees(t_roots[0]))
        ax.plot(degrees(t_roots[0]), y_roots[0, i], ".")
    fig.suptitle("{}:{}".format(
        soln_file.config_filename,
        soln_file.config_input.label
    ))
    return fig


def generate_plot_combine(soln_file, **kwargs):
    """
    Generate plot, with enough freedom to be able to format fig.
    Combine velocities, fields onto on plot
    """
    soln_range = kwargs.pop("soln_range", "0")
    soln_instance = get_solutions(soln_file, soln_range)
    soln = soln_instance.solution
    angles = soln_instance.angles
    cons = soln_instance.initial_conditions
    inp = soln_instance.solution_input

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
    stop = float(kwargs.pop("stop", 90))

    plot_props = OrderedDict([
        ("velocity", {
            "normalisation": v_norm / KM,  # km/s
            "y_label": "Velocity Field (km/s)",
            "lines": [
                {
                    "label": "v_r",
                    "index": 3,
                },
                {
                    "label": "v_φ",
                    "index": 4,
                    "offset": sqrt(cons.norm_kepler_sq) * v_norm / KM
                },
                {
                    "label": "v_θ",
                    "index": 5,
                },
            ],
        }),
        ("density", {
            "y_label": "Density ($g cm^{-3}$)",
            "normalisation": ρ_norm,
            "scale": "log",
            "lines": [
                {
                    "label": "ρ",
                    "index": 6,
                }
            ],
        }),
        ("fields", {
            "normalisation": B_norm,
            "y_label": "Magnetic Field (G)",
            "lines": [
                {
                    "label": "B_r",
                    "index": 0,
                },
                {
                    "label": "B_φ",
                    "index": 1,
                },
                {
                    "label": "B_θ",
                    "index": 2,
                },
            ],
        }),
    ])

    if with_slow:
        plot_props["velocity"]["lines"].append({
            "label": "slow",
            "data": wave_speeds[MHD_WAVE_INDEX["slow"]],
        })
    if with_alfven:
        plot_props["velocity"]["lines"].append({
            "label": "alfven",
            "data": wave_speeds[MHD_WAVE_INDEX["alfven"]],
        })
    if with_fast:
        plot_props["velocity"]["lines"].append({
            "label": "fast",
            "data": wave_speeds[MHD_WAVE_INDEX["fast"]],
        })
    if with_sonic:
        plot_props["velocity"]["lines"].append({
            "label": "sound",
            "data": np.ones(len(soln)),
        })

    indexes = degrees(angles) <= stop

    fig, axes = plt.subplots(
        nrows=3, ncols=1, tight_layout=True, sharex=True,
        gridspec_kw=dict(hspace=0),
        **kwargs
    )
    axes.shape = len(plot_props)
    fig.suptitle("{}:{}".format(
        soln_file.config_filename,
        soln_file.config_input.label
    ))
    for i, plot_name in enumerate(plot_props):
        ax = axes[i]
        settings = plot_props[plot_name]
        for line in settings["lines"]:
            if line.get("index") is not None:
                data = soln[:, line["index"]]
            else:
                data = line["data"]
            ax.plot(
                degrees(angles[indexes]),
                (
                    data[indexes] * settings["normalisation"] -
                    line.get("offset", 0)
                ), linestyle, label=line["label"]
            )
        if i == len(plot_props) - 1:  # label only the bottom one
            ax.set_xlabel("angle from plane (°)")
        if i % 2 == 1:
            ax.tick_params(
                axis='y', which='both', labelleft='off', labelright='on'
            )
        ax.set_ylabel(settings["y_label"])
        ax.set_yscale(settings.get("scale", "linear"))
        ax.legend()
        better_sci_format(ax.yaxis)
    fig.subplots_adjust(hspace=0)
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
    parser.add_argument(
        "--stop", default=90
    )


def get_plot_args(args):
    """
    Parse plot args
    """
    return {
        # "v_θ scale": args.get("v_θ", "linear"),
        "with slow": args.get("with_slow", False),
        "with alfven": args.get("with_alfven", False),
        "with fast": args.get("with_fast", False),
        "with sonic": args.get("with_sonic", False),
        "line style": args.get("line_style", "-"),
        "soln_range": args.get("soln_range", "0"),
        "stop": args.get("stop", "90"),
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
        return soln_file.final_solution
    return soln_file.solutions[soln_range]
