# -*- coding: utf-8 -*-
"""
Plot command for DiscSolver
"""
from numpy import degrees

from .utils import (
    multiple_solution_plotter, analyse_main_wrapper_multisolution,
    analysis_func_wrapper_multisolution, common_plotting_options,
    get_common_plot_args, plot_output_wrapper, DEFAULT_MPL_STYLE,
    distinct_color_map, AnalysisError,
)

from ..utils import ODEIndex


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
    stop=90, figargs=None, title=None, close=True,
    mpl_style=DEFAULT_MPL_STYLE, with_version=True, num_solutions=None
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
    fig, solutions, *, linestyle='-', start=0, stop=90, num_solutions
):
    """
    Generate plot, with enough freedom to be able to format fig
    """
    ax = fig.subplots()

    ax.set_xlabel("angle from plane (°)")

    colors = distinct_color_map(num_solutions)

    for (soln_name, soln), color in zip(solutions, colors):

        internal_data = soln.internal_data
        if internal_data is None:
            raise AnalysisError(
                "Internal data required in {} to generate plot".format(
                    soln_name
                )
            )

        angles = internal_data.angles
        derivs = internal_data.derivs
        indexes = (start <= degrees(angles)) & (degrees(angles) <= stop)
        ax.plot(
            degrees(angles[indexes]),
            derivs[indexes, ODEIndex.v_θ],
            linestyle, label=soln_name, color=color,
        )
        ax.set_ylabel("$v_θ'$")

    ax.legend(loc=0)
    return fig
