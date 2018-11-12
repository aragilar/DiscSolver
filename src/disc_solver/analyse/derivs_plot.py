# -*- coding: utf-8 -*-
"""
Deriv-plot command for DiscSolver
"""
import numpy as np
from numpy import degrees

from .utils import (
    single_solution_plotter, common_plotting_options, analyse_main_wrapper,
    get_common_plot_args, analysis_func_wrapper, plot_output_wrapper,
    AnalysisError
)


def plot_parser(parser):
    """
    Add arguments for plot command to parser
    """
    common_plotting_options(parser)
    parser.add_argument("--nolog", action='store_true', default=False)
    return parser


def get_plot_args(args):
    """
    Parse plot args
    """
    return {
        "nolog": args.get("nolog", False),
    }


@analyse_main_wrapper(
    "Plot derivs for DiscSolver",
    plot_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
        "plot_args": get_plot_args,
    }
)
def derivs_main(soln, *, soln_range, common_plot_args, plot_args):
    """
    Entry point for ds-derivs-plot
    """
    return derivs_plot(
        soln, soln_range=soln_range, **common_plot_args, **plot_args
    )


@analysis_func_wrapper
def derivs_plot(
    soln, *, soln_range=None, plot_filename=None, show=False, stop=90,
    figargs=None, linestyle='.', title=None, nolog=False, close=True, filename
):
    """
    Show derivatives
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_derivs_plot(
        soln, soln_range, linestyle=linestyle, stop=stop, figargs=figargs,
        title=title, nolog=nolog, filename=filename,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@single_solution_plotter
def generate_derivs_plot(
    fig, soln, *, linestyle='.', stop=90, nolog=False
):
    """
    Generate plot of derivatives
    """
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
    if internal_data is None:
        raise AnalysisError("Internal data required to generate plot")
    deriv_angles = internal_data.angles
    derivs = internal_data.derivs
    npnot = np.logical_not
    npand = np.logical_and
    indexes = degrees(deriv_angles) <= stop

    axes = fig.subplots(nrows=2, ncols=4, sharex=True)
    axes.shape = len(param_names)
    for i, settings in enumerate(param_names):
        ax = axes[i]
        if nolog:
            ax.plot(
                degrees(deriv_angles[indexes]),
                derivs[indexes, i], linestyle,
            )

        else:
            pos_deriv = derivs[:, i] >= 0
            ax.plot(
                degrees(deriv_angles[npand(pos_deriv, indexes)]),
                derivs[npand(pos_deriv, indexes), i], linestyle + "b",
            )
            ax.plot(
                degrees(deriv_angles[npand(npnot(pos_deriv), indexes)]),
                - derivs[npand(npnot(pos_deriv), indexes), i], linestyle + "g",
            )
            ax.set_yscale(settings.get("scale", "log"))

        ax.set_xlabel("angle from plane (°)")
        ax.set_title(settings["name"])
        if settings.get("legend"):
            ax.legend()
    return fig
