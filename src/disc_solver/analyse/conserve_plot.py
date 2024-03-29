# -*- coding: utf-8 -*-
"""
Conserve-plot command for DiscSolver
"""
from numpy import degrees
import matplotlib.pyplot as plt
from matplotlib.style import context as use_style

from .utils import (
    plot_output_wrapper, common_plotting_options, get_common_plot_args,
    distinct_color_map, analyse_multisolution_wrapper, DEFAULT_MPL_STYLE,
    add_version_to_plot,
)
from ..utils import ODEIndex

NUM_ITEMS_PER_LEGEND_COLUMN = 15
CLI_DESCRIPTION = "Show conservence in v_θ of solutions"


def conserve_options(parser):
    """
    Add arguments for conserve command to parser
    """
    common_plotting_options(parser)
    return parser


def get_plot_args(args):
    """
    Parse plot args
    """
    # pylint: disable=unused-argument
    return {}


@analyse_multisolution_wrapper(
    "Show divergence in v_θ of solutions",
    conserve_options,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
        "plot_args": get_plot_args,
    }
)
def conserve_main(*, solutions, common_plot_args, plot_args, filename):
    """
    Entry point for ds-diverge-plot
    """
    return conserve_plot(
        solutions, **plot_args, **common_plot_args, filename=filename,
    )


def generate_conserve_plot(
    solutions, *, figargs=None, start=0, stop=90, linestyle='-'
):
    """
    Generate plot to compare how different runs change in v_θ
    """
    if figargs is None:
        figargs = {}
    fig, ax = plt.subplots(**figargs)

    colors = distinct_color_map(len(solutions))

    for (soln_name, soln), color in zip(solutions, colors):
        solution = soln.solution
        angles = soln.angles
        indexes = (start <= degrees(angles)) & (degrees(angles) <= stop)
        ax.plot(
            degrees(angles[indexes]),
            solution[indexes, ODEIndex.ρ] * solution[indexes, ODEIndex.v_θ],
            label=soln_name, color=color, linestyle=linestyle,
        )

    ax.set_xlabel("θ — angle from plane (°)")
    ax.set_ylabel("$ρ v_θ / c_s$")
    ax.legend(ncol=max(1, len(solutions)//NUM_ITEMS_PER_LEGEND_COLUMN))
    return fig


def conserve_plot(
    solutions, *, plot_filename=None, show=False, start=0, stop=90,
    figargs=None, title=None, linestyle='-', close=True,
    mpl_style=DEFAULT_MPL_STYLE, use_E_r=False, with_version=True, filename,
):
    """
    Plot solution to file, with velocities, fields onto on one plot
    """
    # pylint: disable=unused-argument
    with use_style(mpl_style):
        fig = generate_conserve_plot(
            solutions, start=start, stop=stop, figargs=figargs,
            linestyle=linestyle,
        )
        if title is None:
            fig.suptitle(f"Conservation of $ρ v_θ$ for {filename}")
        else:
            fig.suptitle(title)

        if with_version:
            add_version_to_plot(fig)

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )
