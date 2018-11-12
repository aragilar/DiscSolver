# -*- coding: utf-8 -*-
"""
Plot command for DiscSolver
"""
from numpy import degrees, sqrt

from .utils import (
    multiple_solution_plotter, analyse_main_wrapper_multisolution,
    analysis_func_wrapper_multisolution, common_plotting_options,
    get_common_plot_args, plot_output_wrapper,
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
def plot_main(solns, *, common_plot_args, plot_args):
    """
    Entry point for ds-plot
    """
    return compare_plot(solns, **common_plot_args, **plot_args)


@analysis_func_wrapper_multisolution
def compare_plot(
    solns, *, plot_filename=None, show=False, linestyle='-', stop=90,
    figargs=None, v_θ_scale="linear", title=None, close=True
):
    """
    Plot solutions to file
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    # pylint: disable=no-value-for-parameter
    fig = generate_plot(
        solns, linestyle=linestyle, stop=stop, figargs=figargs,
        v_θ_scale=v_θ_scale, title=title,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@multiple_solution_plotter
def generate_plot(fig, solns, *, linestyle='-', stop=90, v_θ_scale="linear"):
    """
    Generate plot, with enough freedom to be able to format fig
    """
    axes = fig.subplots(
        nrows=2, ncols=4, sharex=True, gridspec_kw=dict(hspace=0)
    )

    # only add label to bottom plots
    for ax in axes[1]:
        ax.set_xlabel("angle from plane (°)")
    axes = axes.flatten()

    for id_num, soln in enumerate(solns):
        solution = soln.solution
        angles = soln.angles
        cons = soln.initial_conditions

        indexes = degrees(angles) <= stop

        param_names = [
            {
                "name": "$B_r/B_0$",
            },
            {
                "name": "$B_φ/B_0$",
            },
            {
                "name": "$B_θ/B_0$",
            },
            {
                "name": "$v_r/c_s$",
            },
            {
                "name": "$(v_φ - v_k)/c_s$",
                "offset": sqrt(cons.norm_kepler_sq)
            },
            {
                "name": "$v_θ/c_s$",
                "legend": True,
                "scale": v_θ_scale,
            },
            {
                "name": "$ρ/ρ_0$",
                "scale": "log",
            },
            {
                "name": "$B_φ'/B_0$",
            },
        ]

        for i, settings in enumerate(param_names):
            ax = axes[i]
            ax.plot(
                degrees(angles[indexes]),
                (
                    solution[:, i] - settings.get("offset", 0)
                )[indexes], linestyle, label=str(id_num)
            )
            ax.set_ylabel(settings["name"])
            ax.set_yscale(settings.get("scale", "linear"))
    for ax in axes:
        ax.legend(loc=0)
    return fig
