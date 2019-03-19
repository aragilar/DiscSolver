# -*- coding: utf-8 -*-
"""
Plot command for DiscSolver
"""
from numpy import degrees

from .utils import (
    multiple_solution_plotter, analyse_main_wrapper_multisolution,
    analysis_func_wrapper_multisolution, common_plotting_options,
    get_common_plot_args, plot_output_wrapper, DEFAULT_MPL_STYLE,
    get_common_arguments, PlotOrdering, distinct_color_map,
    single_fig_legend_setup,
)


def plot_parser(parser):
    """
    Add arguments for plot command to parser
    """
    common_plotting_options(parser)
    parser.add_argument("--v_θ", choices=("log", "linear"), default="linear")
    return parser


def get_plot_args(args):
    """
    Parse plot args
    """
    return {
        "v_θ_scale": args.get("v_θ", "linear"),
    }


@analyse_main_wrapper_multisolution(
    "Main plotter for DiscSolver",
    plot_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
        "plot_args": get_plot_args,
    }
)
def plot_main(solutions, *, common_plot_args, plot_args):
    """
    Entry point for ds-plot
    """
    return compare_plot(solutions, **common_plot_args, **plot_args)


@analysis_func_wrapper_multisolution
def compare_plot(
    solutions, *, plot_filename=None, show=False, linestyle='-', start=0,
    stop=90, figargs=None, v_θ_scale="linear", title=None, close=True,
    mpl_style=DEFAULT_MPL_STYLE, with_version=True, num_solutions=None
):
    """
    Plot solutions to file
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    # pylint: disable=no-value-for-parameter
    fig = generate_plot(
        solutions, linestyle=linestyle, start=start, stop=stop,
        figargs=figargs, v_θ_scale=v_θ_scale, title=title, mpl_style=mpl_style,
        with_version=with_version, num_solutions=num_solutions,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@multiple_solution_plotter
def generate_plot(
    fig, solutions, *, linestyle='-', start=0, stop=90, v_θ_scale="linear",
    use_E_r=False, num_solutions
):
    """
    Generate plot, with enough freedom to be able to format fig
    """
    axes, figlegend = single_fig_legend_setup(
        fig, nrows=2, ncols=4, sharex=True, gridspec_kw=dict(hspace=0)
    )
    # only add label to bottom plots
    for ax in axes[1]:
        ax.set_xlabel("angle from plane (°)")
    axes = axes.flatten()

    colors = distinct_color_map(num_solutions)

    for (soln_name, soln), color in zip(solutions, colors):
        solution = soln.solution
        angles = soln.angles
        cons = soln.initial_conditions

        indexes = (start <= degrees(angles)) & (degrees(angles) <= stop)

        ordering = PlotOrdering.E_r if use_E_r else PlotOrdering.B_φ_prime

        param_names = get_common_arguments(
            ordering, v_θ_scale=v_θ_scale, initial_conditions=cons
        )

        for i, settings in enumerate(param_names):
            ax = axes[i]
            ax.plot(
                degrees(angles[indexes]),
                (
                    solution[:, i] - settings.get("offset", 0)
                )[indexes], linestyle, label=soln_name, color=color,
            )
            ax.set_ylabel(settings["name"])
            ax.set_yscale(settings.get("scale", "linear"))
    figlegend(fontsize="small")
    return fig
