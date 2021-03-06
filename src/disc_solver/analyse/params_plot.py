# -*- coding: utf-8 -*-
"""
Params-plot command for DiscSolver
"""
from numpy import degrees

from .utils import (
    single_solution_plotter, common_plotting_options, analyse_main_wrapper,
    get_common_plot_args, analysis_func_wrapper, plot_output_wrapper,
    AnalysisError, DEFAULT_MPL_STYLE, get_common_arguments, PlotOrdering,
    plot_log_lines,
)


def plot_parser(parser):
    """
    Add arguments for params-plot command to parser
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
def params_main(soln, *, soln_range, common_plot_args):
    """
    Entry point for ds-params-plot
    """
    return params_plot(soln, soln_range=soln_range, **common_plot_args)


@analysis_func_wrapper
def params_plot(
    soln, *, soln_range=None, plot_filename=None, show=False, start=0, stop=90,
    figargs=None, linestyle='.', title=None, close=True, filename,
    mpl_style=DEFAULT_MPL_STYLE, with_version=True
):
    """
    Show solution at every step the solver takes.
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_params_plot(
        soln, soln_range, linestyle=linestyle, start=start, stop=stop,
        figargs=figargs, title=title, filename=filename, mpl_style=mpl_style,
        with_version=with_version,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@single_solution_plotter
def generate_params_plot(
    fig, soln, *, linestyle='.', start=0, stop=90, use_E_r=False
):
    """
    Generate plot of all values, including intermediate values
    """
    if use_E_r:
        raise AnalysisError("Function needs modification to work with use_E_r")

    internal_data = soln.internal_data
    if internal_data is None:
        raise AnalysisError("Internal data required to generate plot")

    ordering = PlotOrdering.E_r if use_E_r else PlotOrdering.B_φ_prime

    cons = soln.initial_conditions
    param_names = get_common_arguments(ordering, initial_conditions=cons)

    param_angles = internal_data.angles
    params = internal_data.params
    indexes = (
        (start <= degrees(param_angles)) & (degrees(param_angles) <= stop)
    )

    axes = fig.subplots(nrows=2, ncols=4, sharex=True)
    axes.shape = len(param_names)
    for i, settings in enumerate(param_names):
        ax = axes[i]
        plot_log_lines(
            ax, degrees(param_angles[indexes]), params[indexes, i], linestyle
        )
        ax.set_xlabel("angle from plane (°)")
        ax.set_title(settings["name"])
    return fig
