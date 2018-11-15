# -*- coding: utf-8 -*-
"""
Check-taylor command for DiscSolver
"""
import numpy as np
from numpy import degrees

from .utils import (
    single_solution_plotter, common_plotting_options, analyse_main_wrapper,
    get_common_plot_args, analysis_func_wrapper, plot_output_wrapper,
    AnalysisError, DEFAULT_MPL_STYLE,
)


def check_taylor_parser(parser):
    """
    Add arguments for check-taylor command to parser
    """
    common_plotting_options(parser)
    parser.add_argument(
        "--show-values", action="store_true", default=False
    )
    return parser


def get_plot_args(args):
    """
    Parse plot args
    """
    return {
        "show_values": args.get("show_values", False),
    }


@analyse_main_wrapper(
    "Plot derivs for DiscSolver",
    check_taylor_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
        "plot_args": get_plot_args,
    }
)
def taylor_main(soln, *, soln_range, plot_args, common_plot_args):
    """
    Entry point for ds-params-plot
    """
    return taylor_plot(
        soln, soln_range=soln_range, **plot_args,
        **common_plot_args
    )


@analysis_func_wrapper
def taylor_plot(
    soln, *, soln_range=None, plot_filename=None, show=False, stop=90,
    figargs=None, linestyle='.', title=None, show_values=False, close=True,
    filename, mpl_style=DEFAULT_MPL_STYLE
):
    """
    Show solution at every step the solver takes.
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_taylor_plot(
        soln, soln_range, linestyle=linestyle, stop=stop, figargs=figargs,
        title=title, show_values=show_values, filename=filename,
        mpl_style=mpl_style,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@single_solution_plotter
def generate_taylor_plot(
    fig, soln, *, show_values=False, stop=90, linestyle='.', use_E_r=False
):
    # pylint: disable=unused-variable,unused-argument
    """
    Compare derivatives from taylor series to full version
    """
    if soln.internal_data is None:
        raise AnalysisError("Internal data required to generate plot")

    v_r_normal = soln.internal_data.v_r_normal
    v_φ_normal = soln.internal_data.v_φ_normal
    ρ_normal = soln.internal_data.ρ_normal
    v_r_taylor = soln.internal_data.v_r_taylor
    v_φ_taylor = soln.internal_data.v_φ_taylor
    ρ_taylor = soln.internal_data.ρ_taylor

    deriv_angles = degrees(soln.internal_data.angles)
    axes = fig.subplots(ncols=3)
    if show_values:
        axes[0].plot(
            deriv_angles[v_r_normal >= 0],
            v_r_normal[v_r_normal >= 0],
            'b.'
        )
        axes[0].plot(
            deriv_angles[v_r_normal < 0],
            - v_r_normal[v_r_normal < 0],
            'r.'
        )
        axes[0].plot(
            deriv_angles[v_r_taylor >= 0],
            v_r_taylor[v_r_taylor >= 0],
            'bx'
        )
        axes[0].plot(
            deriv_angles[v_r_taylor < 0],
            - v_r_taylor[v_r_taylor < 0],
            'rx'
        )
        axes[1].plot(
            deriv_angles[v_φ_normal >= 0],
            v_φ_normal[v_φ_normal >= 0],
            'b.'
        )
        axes[1].plot(
            deriv_angles[v_φ_normal < 0],
            - v_φ_normal[v_φ_normal < 0],
            'r.'
        )
        axes[1].plot(
            deriv_angles[v_φ_taylor >= 0],
            v_φ_taylor[v_φ_taylor >= 0],
            'bx'
        )
        axes[1].plot(
            deriv_angles[v_φ_taylor < 0],
            - v_φ_taylor[v_φ_taylor < 0],
            'rx'
        )
        axes[2].plot(
            deriv_angles[ρ_normal >= 0],
            ρ_normal[ρ_normal >= 0],
            'b.'
        )
        axes[2].plot(
            deriv_angles[ρ_normal < 0],
            - ρ_normal[ρ_normal < 0],
            'r.'
        )
        axes[2].plot(
            deriv_angles[ρ_taylor >= 0],
            ρ_taylor[ρ_taylor >= 0],
            'bx'
        )
        axes[2].plot(
            deriv_angles[ρ_taylor < 0],
            - ρ_taylor[ρ_taylor < 0],
            'rx'
        )
    else:
        axes[0].plot(
            degrees(deriv_angles),
            np.abs(v_r_normal / v_r_taylor), '.'
        )
        axes[1].plot(
            degrees(deriv_angles),
            np.abs(v_φ_normal / v_φ_taylor), '.'
        )
        axes[2].plot(
            degrees(deriv_angles),
            np.abs(ρ_normal / ρ_taylor), '.'
        )

    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")

    axes[0].set_title("v_r")
    axes[1].set_title("v_φ")
    axes[2].set_title("ρ")

    return fig
