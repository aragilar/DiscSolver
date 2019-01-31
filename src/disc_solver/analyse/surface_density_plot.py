# -*- coding: utf-8 -*-
"""
surface-density-plot command for DiscSolver
"""
from numpy import degrees, cumsum

from ..utils import ODEIndex
from .utils import (
    single_solution_plotter, common_plotting_options, analyse_main_wrapper,
    get_common_plot_args, analysis_func_wrapper, plot_output_wrapper,
    DEFAULT_MPL_STYLE,
)


def plot_parser(parser):
    """
    Add arguments for surface-density-plot command to parser
    """
    common_plotting_options(parser)
    return parser


@analyse_main_wrapper(
    "Plot surface density for DiscSolver",
    plot_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
    }
)
def surface_density_main(soln, *, soln_range, common_plot_args):
    """
    Entry point for ds-surface-density-plot
    """
    return surface_density_plot(
        soln, soln_range=soln_range, **common_plot_args
    )


@analysis_func_wrapper
def surface_density_plot(
    soln, *, soln_range=None, plot_filename=None, show=False, start=0, stop=90,
    figargs=None, linestyle='.', title=None, close=True, filename,
    mpl_style=DEFAULT_MPL_STYLE, with_version=True
):
    """
    Show derivatives
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_surface_density_plot(
        soln, soln_range, linestyle=linestyle, start=start, stop=stop,
        figargs=figargs, title=title, filename=filename, mpl_style=mpl_style,
        with_version=with_version,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@single_solution_plotter
def generate_surface_density_plot(
    fig, soln, *, linestyle='.', start=0, stop=90, use_E_r=False
):
    """
    Friendlier plot for talks
    """
    # pylint: disable=unused-variable,unused-argument

    solution = soln.solution
    angles = soln.angles

    indexes = (start <= degrees(angles)) & (degrees(angles) <= stop)

    cum_σ = cumsum(solution[:, ODEIndex.ρ])
    axes = fig.subplots()
    axes.plot(angles[indexes], cum_σ[indexes])
    axes.set_xlabel("angle from plane (°)")
    axes.set_ylabel("σ")
    return fig
