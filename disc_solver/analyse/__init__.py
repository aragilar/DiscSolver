# -*- coding: utf-8 -*-
"""
Stuff to analyse solutions
"""

from math import pi, sqrt

import numpy as np
import matplotlib.pyplot as plt

from ..constants import KM
from ..utils import better_sci_format, mhd_wave_speeds, MHD_WAVE_INDEX


def generate_plot(angles, soln, inp, cons, **kwargs):
    """
    Generate plot, with enough freedom to be able to format fig
    """
    B_norm, v_norm, ρ_norm = cons.B_norm, cons.v_norm, cons.ρ_norm
    zero_soln = np.zeros(len(soln))
    v = np.array([zero_soln, zero_soln, soln[:, 5]])
    wave_speeds = np.sqrt(mhd_wave_speeds(
        v.T, soln[:, 0:3], soln[:, 6], inp.c_s
    ))
    linestyle = kwargs.pop("line style")
    with_slow = kwargs.pop("with slow")
    with_alfven = kwargs.pop("with alfven")
    with_fast = kwargs.pop("with fast")

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

    fig, axes = plt.subplots(
        nrows=2, ncols=4, tight_layout=True, sharex=True, **kwargs
    )
    axes.shape = len(param_names)
    for i, settings in enumerate(param_names):
        ax = axes[i]
        ax.plot(
            90 - (angles * 180 / pi),
            (
                soln[:, i] * settings["normalisation"] -
                settings.get("offset", 0)
            ), linestyle,
        )
        for extra in settings.get("extras", []):
            ax.plot(
                90 - (angles * 180 / pi),
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


def get_plot_args(args):
    """
    Parse plot args
    """
    return {
        "v_θ scale": args.get("v_θ", "linear"),
        "with slow": args.get("with_slow", False),
        "with alfven": args.get("with_alfven", False),
        "with fast": args.get("with_fast", False),
        "line style": args.get("line_style", "-")
    }


def generate_deriv_plot(angles, soln, inp, cons, **kwargs):
    """
    Generate plot of derivatives
    """
    # pylint: disable=unused-argument
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
    internal_data = kwargs.pop("internal_data")
    deriv_angles = np.array(internal_data["angles"])
    derivs = np.array(internal_data["derivs"])
    npnot = np.logical_not

    fig, axes = plt.subplots(
        nrows=2, ncols=4, tight_layout=True, sharex=True, **kwargs
    )
    axes.shape = len(param_names)
    for i, settings in enumerate(param_names):
        ax = axes[i]
        pos_deriv = derivs[:, i] >= 0
        ax.plot(
            90 - (deriv_angles[pos_deriv] * 180 / pi),
            derivs[pos_deriv, i], linestyle,
        )
        ax.plot(
            90 - (deriv_angles[npnot(pos_deriv)] * 180 / pi),
            - derivs[npnot(pos_deriv), i], linestyle,
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
        "internal_data": args["internal_data"]
    }


def generate_params_plot(angles, soln, inp, cons, **kwargs):
    """
    Generate plot of all values, including intermediate values
    """
    # pylint: disable=unused-argument
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
    internal_data = kwargs.pop("internal_data")
    param_angles = np.array(internal_data["angles"])
    params = np.array(internal_data["params"])
    npnot = np.logical_not

    fig, axes = plt.subplots(
        nrows=2, ncols=4, tight_layout=True, sharex=True, **kwargs
    )
    axes.shape = len(param_names)
    for i, settings in enumerate(param_names):
        ax = axes[i]
        pos_params = params[:, i] >= 0
        ax.plot(
            90 - (param_angles[pos_params] * 180 / pi),
            params[pos_params, i], linestyle,
        )
        ax.plot(
            90 - (param_angles[npnot(pos_params)] * 180 / pi),
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
        "internal_data": args["internal_data"]
    }
