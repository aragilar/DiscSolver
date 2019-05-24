# -*- coding: utf-8 -*-
"""
Plot command for DiscSolver
"""
from numpy import degrees, sum as np_sum
from scipy.integrate import simps

from ..float_handling import float_type
from ..utils import ODEIndex

from .utils import (
    multiple_solution_plotter, analyse_main_wrapper_multisolution,
    analysis_func_wrapper_multisolution, common_plotting_options,
    get_common_plot_args, plot_output_wrapper, DEFAULT_MPL_STYLE,
)

MARKER_LIST = [
    '.', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h',
    'H', '+', 'x', 'D', 'd', 'P', 'X',
]

TAB_COLOR_MAP = {
    'C0': 'blue',
    'C1': 'orange',
    'C2': 'green',
    'C3': 'red',
    'C4': 'purple',
    'C5': 'brown',
    'C6': 'pink',
    'C7': 'gray',
    'C8': 'olive',
    'C9': 'cyan',
}


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
    axes = fig.subplots(ncols=2)
    for ax in axes:
        ax.set_xlabel("$η_A/(c_{s}^2 Ω)$")
    axes[0].set_ylabel("$B_z^2/Σ$")  # B_z at midplane is 1
    axes[1].set_ylabel("$\\dot{M}_{out}/\\dot{M}_{in}$")
    axes[1].set_yscale("log")

    axes[0].set_facecolor('none')
    axes[1].set_facecolor('none')

    v_r_count = 0
    v_r_mapping = {}
    a_0_count = 0
    a_0_mapping = {}

    for soln_name, soln in solutions:
        angles = soln.angles
        η_A = soln.initial_conditions.η_A
        a_0 = soln.initial_conditions.a_0
        v_r = soln.initial_conditions.init_con[ODEIndex.v_r]
        if v_r not in v_r_mapping:
            v_r_mapping[v_r] = 'C' + str(v_r_count)
            v_r_count = (v_r_count + 1) % 10
        if a_0 not in a_0_mapping:
            a_0_mapping[a_0] = MARKER_LIST[a_0_count]
            a_0_count = (a_0_count + 1) % len(MARKER_LIST)

        m_color = v_r_mapping[v_r]
        m_marker = a_0_mapping[a_0]

        indexes = (start <= degrees(angles)) & (degrees(angles) <= stop)
        Σ = compute_Σ(soln, indexes)
        c_s_on_v_k = soln.solution_input.c_s_on_v_k

        axes[0].plot(
            η_A / c_s_on_v_k, a_0/Σ, marker=m_marker, color=m_color,
            label=soln_name
        )
        axes[1].plot(
            η_A / c_s_on_v_k, compute_M_dot_out_on_M_dot_in(soln, indexes),
            marker=m_marker, color=m_color, label=soln_name,
        )

    for v_r, color in v_r_mapping.items():
        print("v_r = {} -> {} -> {}".format(v_r, color, TAB_COLOR_MAP[color]))

    for a_0, marker in a_0_mapping.items():
        print("a_0 = {} -> {}".format(a_0, marker))

    return fig


def compute_M_dot_out_on_M_dot_in(
    soln, indexes=slice(None), r_in=0.1, r_out=100
):
    """
    Find M_dot_out / M_dot_in
    """
    # Integers don't work with negative powers
    r_in, r_out = float_type(r_in), float_type(r_out)

    solution = soln.solution[indexes]
    angles = soln.angles[indexes]
    γ = soln.initial_conditions.γ
    ρ = solution[:, ODEIndex.ρ]
    v_r = solution[:, ODEIndex.v_r]

    # need minus sign as by default in is negative
    M_in = - simps(y=ρ*v_r, x=angles)

    scaling = (r_out ** (2 * γ) - r_in ** (2 * γ)) / (2 * γ) / (
        r_in ** (2 * γ - 1) - r_out ** (2 * γ - 1)
    )

    return scaling * (
        solution[-1, ODEIndex.ρ] * solution[-1, ODEIndex.v_θ]
    ) / M_in


def compute_Σ(soln, indexes=slice(None)):
    """
    Compute Σ
    """
    return np_sum(soln.solution[indexes, ODEIndex.ρ])
