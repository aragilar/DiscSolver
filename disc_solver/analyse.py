# -*- coding: utf-8 -*-
"""
Stuff to analyse solutions
"""

from math import pi, sqrt

import numpy as np
import matplotlib.pyplot as plt

from .constants import KM
from .utils import better_sci_format, mhd_wave_speeds, MHD_WAVE_INDEX


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
        },
        {
            "name": "v_θ",
            "y_label": "Velocity Field (km/s)",
            "normalisation": v_norm / KM,  # km/s
            "scale": "log",
            "legend": True,
            "extras": [
                {
                    "label": "slow",
                    "data": wave_speeds[MHD_WAVE_INDEX["slow"]],
                    "normalisation": v_norm / KM,
                },
                {
                    "label": "alfven",
                    "data": wave_speeds[MHD_WAVE_INDEX["alfven"]],
                    "normalisation": v_norm / KM,
                },
                {
                    "label": "fast",
                    "data": wave_speeds[MHD_WAVE_INDEX["fast"]],
                    "normalisation": v_norm / KM,
                },
            ]
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

    fig, axes = plt.subplots(
        nrows=2, ncols=4, tight_layout=True, sharex=True, **kwargs
    )
    axes.shape = len(param_names)
    for i, settings in enumerate(param_names):
        ax = axes[i]
        ax.plot(
            90 - (angles * 180 / pi),
            soln[:, i] * settings["normalisation"]
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


def info(inp, cons):
    """
    Output info about the solution
    """
    print("input settings:")
    for name, value in vars(inp).items():
        print(" {: <20}: {}".format(name, value))
    print("initial conditions:")
    for name, value in vars(cons).items():
        print(" {: <20}: {}".format(name, value))
    print("other info: ")
    print("v_a/c_s at midplane:", sqrt(inp.B_θ**2 / (4*pi*inp.ρ)) / inp.c_s)
