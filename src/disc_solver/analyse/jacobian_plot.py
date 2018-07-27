# -*- coding: utf-8 -*-
"""
Jacobian-plot command for DiscSolver
"""
import numpy as np
from numpy import degrees
import matplotlib.pyplot as plt
from scipy.linalg import eigvals

from ..utils import ODEIndex
from .utils import (
    single_solution_plotter, common_plotting_options, analyse_main_wrapper,
    get_common_plot_args, analysis_func_wrapper, plot_output_wrapper,
)


def plot_parser(parser):
    """
    Add arguments for plot command to parser
    """
    common_plotting_options(parser)
    return parser


@analyse_main_wrapper(
    "Plot jacobian for DiscSolver",
    plot_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
    }
)
def jacobian_main(soln, *, soln_range, common_plot_args):
    """
    Entry point for ds-jacobian-plot
    """
    return jacobian_plot(soln, soln_range=soln_range, **common_plot_args)


@analysis_func_wrapper
def jacobian_plot(
    soln, *, soln_range=None, plot_filename=None, show=False, stop=90,
    figargs=None, linestyle='.', title=None, close=True
):
    """
    Show jacobian
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_jacobian_plot(
        soln, soln_range, linestyle=linestyle, stop=stop, figargs=figargs,
        title=title,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@single_solution_plotter
def generate_jacobian_plot(
    soln, *, linestyle='.', figargs=None, stop=90
):
    """
    Generate plot of jacobians
    """
    if figargs is None:
        figargs = {}

    jacobian_data = soln.internal_data.jacobian_data
    angles = jacobian_data.angles
    jacobians = jacobian_data.jacobians
    npnot = np.logical_not
    npand = np.logical_and
    indexes = degrees(angles) <= stop

    data = np.array([eigvals(j) for j in jacobians])

    fig, axes = plt.subplots(
        nrows=2, ncols=6, constrained_layout=True, sharex=True, **figargs
    )
    axes.shape = 12
    for param in ODEIndex:
        ax = axes[param]
        pos_data = data[:, param] >= 0
        ax.plot(
            degrees(angles[npand(pos_data, indexes)]),
            data[npand(pos_data, indexes), param], linestyle + "b",
        )
        ax.plot(
            degrees(angles[npand(npnot(pos_data), indexes)]),
            - data[npand(npnot(pos_data), indexes), param], linestyle + "g",
        )
        ax.set_xlabel("angle from plane (°)")
        ax.set_yscale("log")
        ax.set_title(param.name)
    return fig
