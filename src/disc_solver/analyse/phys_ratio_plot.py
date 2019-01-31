# -*- coding: utf-8 -*-
"""
Plot command for DiscSolver
"""
from numpy import degrees, sum as np_sum

from ..utils import ODEIndex

from .utils import (
    multiple_solution_plotter, analyse_main_wrapper_multisolution,
    analysis_func_wrapper_multisolution, common_plotting_options,
    get_common_plot_args, plot_output_wrapper, DEFAULT_MPL_STYLE,
)


def plot_parser(parser):
    """
    Add arguments for plot command to parser
    """
    common_plotting_options(parser)
    return parser


@analyse_main_wrapper_multisolution(
    "Main plotter for DiscSolver",
    plot_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
    }
)
def plot_main(solutions, *, common_plot_args):
    """
    Entry point for ds-plot
    """
    return plot(solutions, **common_plot_args)


@analysis_func_wrapper_multisolution
def plot(
    solutions, *, plot_filename=None, show=False, linestyle='-', start=0,
    stop=90, figargs=None, title=None, close=True, mpl_style=DEFAULT_MPL_STYLE,
    with_version=True, num_solutions=None
):
    """
    Plot solutions to file
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    # pylint: disable=no-value-for-parameter
    fig = generate_plot(
        solutions, linestyle=linestyle, start=start, stop=stop,
        figargs=figargs, title=title, mpl_style=mpl_style,
        with_version=with_version, num_solutions=num_solutions,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@multiple_solution_plotter
def generate_plot(
    fig, solutions, *, num_solutions, linestyle='-', start=0, stop=90
):
    """
    Generate plot, with enough freedom to be able to format fig
    """
    # pylint: disable=unused-argument
    axes = fig.subplots()
    axes.set_xlabel("$η_A$")
    axes.set_ylabel("$B_z^2/σ$")  # B_z at midplane is 1

    for soln_name, soln in solutions:
        solution = soln.solution
        angles = soln.angles
        η_A = soln.initial_conditions.η_A
        a_0 = soln.initial_conditions.a_0

        indexes = (start <= degrees(angles)) & (degrees(angles) <= stop)
        σ = np_sum(solution[indexes, ODEIndex.ρ])

        axes.plot(η_A, a_0/σ, marker='.', color='C0', label=soln_name)

    return fig
