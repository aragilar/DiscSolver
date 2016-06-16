# -*- coding: utf-8 -*-
"""
Deriv-plot command for DiscSolver
"""
import numpy as np
from numpy import degrees
import matplotlib.pyplot as plt

from ..utils import better_sci_format
from .utils import (
    single_solution_plotter, common_plotting_options, analyse_main_wrapper,
    get_common_plot_args
)


def plot_parser(parser):
    """
    Add arguments for plot command to parser
    """
    common_plotting_options(parser)
    return parser


@analyse_main_wrapper(
    "Plot derivs for DiscSolver",
    plot_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
    }
)
def derivs_plot(soln_file, *, plot_args):
    """
    Show derivatives
    """
    generate_derivs_plot(soln_file, **plot_args)
    plt.show()


@single_solution_plotter
def generate_derivs_plot(
    soln, *, linestyle='.', figargs=None
):
    """
    Generate plot of derivatives
    """
    if figargs is None:
        figargs = {}

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

    internal_data = soln.internal_data
    deriv_angles = internal_data.angles
    derivs = internal_data.derivs
    npnot = np.logical_not

    fig, axes = plt.subplots(
        nrows=2, ncols=4, tight_layout=True, sharex=True, **figargs
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
