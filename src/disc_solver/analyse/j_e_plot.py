# -*- coding: utf-8 -*-
"""
Plot J and E command for DiscSolver
"""
from numpy import degrees, tan, sqrt
import matplotlib.pyplot as plt

from ..solve.deriv_funcs import deriv_B_r_func

from ..utils import ODEIndex

from .utils import (
    single_solution_plotter, analyse_main_wrapper, analysis_func_wrapper,
    common_plotting_options, get_common_plot_args, plot_output_wrapper,
)

plt.style.use("bmh")


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
    stop=90, figargs=None, title=None, close=True
):
    """
    Plot solution to file
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_plot(
        soln, soln_range, linestyle=linestyle, stop=stop, figargs=figargs,
        title=title,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@single_solution_plotter
def generate_plot(
    soln, *, linestyle='-', stop=90, figargs=None
):
    """
    Generate plot, with enough freedom to be able to format fig
    """
    if figargs is None:
        figargs = {}

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

    fig, axes = plt.subplots(
        nrows=2, ncols=3, constrained_layout=True, sharex=True,
        gridspec_kw=dict(hspace=0), **figargs
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


def J_func(*, γ, θ, B_θ, B_φ, deriv_B_φ, deriv_B_r):
    """
    Compute the currents
    """
    J_r = B_φ * tan(θ) - deriv_B_φ
    J_θ = - B_φ * (1/4 - γ)
    J_φ = deriv_B_r + B_θ * (1/4 - γ)
    return J_r, J_θ, J_φ


def E_func(*, v_r, v_θ, v_φ, B_r, B_θ, B_φ, J_r, J_θ, J_φ, η_O, η_A, η_H):
    """
    Compute the electric field
    """
    B_mag = sqrt(B_r**2 + B_φ**2 + B_θ**2)
    b_r, b_φ, b_θ = B_r/B_mag, B_φ/B_mag, B_θ/B_mag

    E_r = v_φ * B_θ - v_θ * B_φ - (
        η_O * J_r + η_H * (
            J_φ * b_θ - J_θ * b_φ
        ) - η_A * (
            J_φ * b_r * b_φ + J_θ * b_r * b_θ - J_r * (1 - b_r ** 2)
        )
    )

    E_θ = v_r * B_φ - v_φ * B_r - (
        η_O * J_θ + η_H * (
            J_r * b_φ - J_φ * b_r
        ) - η_A * (
            J_r * b_r * b_θ + J_φ * b_θ * b_φ - J_θ * (1 - b_θ ** 2)
        )
    )

    E_φ = v_θ * B_r - v_r * B_θ - (
        η_O * J_φ + η_H * (
            J_θ * b_r - J_r * b_θ
        ) - η_A * (
            J_θ * b_θ * b_φ + J_r * b_r * b_φ - J_φ * (1 - b_φ ** 2)
        )
    )

    return E_r, E_θ, E_φ
