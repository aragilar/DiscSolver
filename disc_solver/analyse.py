# -*- coding: utf-8 -*-
"""
Stuff to analyse solutions
"""

from math import pi

import matplotlib.pyplot as plt

from .constants import KM


def generate_plot(angles, soln, B_norm, v_norm, ρ_norm, **kwargs):
    """
    Generate plot, with enough freedom to be able to format fig
    """
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

    fig, axes = plt.subplots(nrows=2, ncols=4, tight_layout=True, **kwargs)
    axes.shape = len(param_names)
    for i, settings in enumerate(param_names):
        ax = axes[i]
        ax.plot(
            90 - (angles * 180 / pi),
            soln[:, i] * settings["normalisation"]
        )
        ax.set_xlabel("angle from plane (°)")
        ax.set_ylabel(settings["y_label"])
        ax.set_yscale(settings.get("scale", "linear"))
        ax.set_title(settings["name"])
    return fig
