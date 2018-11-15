# -*- coding: utf-8 -*-
"""
Plot J and E command for DiscSolver
"""
from numpy import degrees

from ..solve.deriv_funcs import deriv_B_r_func
from ..solve.j_e_funcs import J_func, E_func

from ..utils import ODEIndex

from .utils import (
    single_solution_plotter, analyse_main_wrapper, analysis_func_wrapper,
    common_plotting_options, get_common_plot_args, plot_output_wrapper,
    DEFAULT_MPL_STYLE, AnalysisError,
)


def plot_parser(parser):
    """
    Add arguments for plot command to parser
    """
    common_plotting_options(parser)
    return parser


@analyse_main_wrapper(
    "Main J and E plotter for DiscSolver",
    plot_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
    }
)
def j_e_plot_main(soln, *, soln_range, common_plot_args):
    """
    Entry point for ds-plot
    """
    return j_e_plot(
        soln, soln_range=soln_range, **common_plot_args
    )


@analysis_func_wrapper
def j_e_plot(
    soln, *, soln_range=None, plot_filename=None, show=False, linestyle='-',
    stop=90, figargs=None, title=None, close=True, filename,
    mpl_style=DEFAULT_MPL_STYLE
):
    """
    Plot solution to file
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_plot(
        soln, soln_range, linestyle=linestyle, stop=stop, figargs=figargs,
        title=title, filename=filename, mpl_style=mpl_style,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@single_solution_plotter
def generate_plot(fig, soln, *, linestyle='-', stop=90, use_E_r=False):
    """
    Generate plot, with enough freedom to be able to format fig
    """
    if use_E_r:
        raise AnalysisError("Function needs modification to work with use_E_r")
    solution = soln.solution
    angles = soln.angles
    cons = soln.initial_conditions

    indexes = degrees(angles) <= stop

    J_r, J_θ, J_φ = J_func(
        γ=cons.γ, θ=angles, B_θ=solution[:, ODEIndex.B_θ],
        B_φ=solution[:, ODEIndex.B_φ],
        deriv_B_φ=solution[:, ODEIndex.B_φ_prime],
        deriv_B_r=deriv_B_r_func(
            γ=cons.γ, θ=angles,
            v_r=solution[:, ODEIndex.v_r], v_θ=solution[:, ODEIndex.v_θ],
            B_r=solution[:, ODEIndex.B_r], B_θ=solution[:, ODEIndex.B_θ],
            B_φ=solution[:, ODEIndex.B_φ], η_O=solution[:, ODEIndex.η_O],
            η_A=solution[:, ODEIndex.η_A], η_H=solution[:, ODEIndex.η_H],
            deriv_B_φ=solution[:, ODEIndex.B_φ_prime],
        ),
    )
    E_r, E_θ, E_φ = E_func(
        v_r=solution[:, ODEIndex.v_r], v_θ=solution[:, ODEIndex.v_θ],
        v_φ=solution[:, ODEIndex.v_φ], B_r=solution[:, ODEIndex.B_r],
        B_θ=solution[:, ODEIndex.B_θ], B_φ=solution[:, ODEIndex.B_φ],
        J_r=J_r, J_θ=J_θ, J_φ=J_φ, η_O=solution[:, ODEIndex.η_O],
        η_A=solution[:, ODEIndex.η_A], η_H=solution[:, ODEIndex.η_H],
    )

    axes = fig.subplots(
        nrows=2, ncols=3, sharex=True, gridspec_kw=dict(hspace=0),
    )

    # only add label to bottom plots
    for ax in axes[1]:
        ax.set_xlabel("angle from plane (°)")

    ax = axes[0, 0]
    ax.plot(
        degrees(angles[indexes]),
        J_r[indexes],
        linestyle
    )
    ax.set_ylabel("$J_r$")

    ax = axes[0, 1]
    ax.plot(
        degrees(angles[indexes]),
        J_θ[indexes],
        linestyle
    )
    ax.set_ylabel("$J_θ$")

    ax = axes[0, 2]
    ax.plot(
        degrees(angles[indexes]),
        J_φ[indexes],
        linestyle
    )
    ax.set_ylabel("$J_φ$")

    ax = axes[1, 0]
    ax.plot(
        degrees(angles[indexes]),
        E_r[indexes],
        linestyle
    )
    ax.set_ylabel("$E_r$")

    ax = axes[1, 1]
    ax.plot(
        degrees(angles[indexes]),
        E_θ[indexes],
        linestyle
    )
    ax.set_ylabel("$E_θ$")

    ax = axes[1, 2]
    ax.plot(
        degrees(angles[indexes]),
        E_φ[indexes],
        linestyle
    )
    ax.set_ylabel("$E_φ$")

    return fig
