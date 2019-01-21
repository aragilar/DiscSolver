# -*- coding: utf-8 -*-
"""
Plot command for DiscSolver
"""
from numpy import degrees, sqrt

from ..utils import (
    mhd_wave_speeds, MHD_Wave_Index, ODEIndex, MAGNETIC_INDEXES,
)

from .utils import (
    multiple_solution_plotter, analyse_main_wrapper_multisolution,
    analysis_func_wrapper_multisolution, common_plotting_options,
    get_common_plot_args, plot_output_wrapper, DEFAULT_MPL_STYLE,
    distinct_color_map,
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
    axes = fig.subplots(
        ncols=2, sharex=True, gridspec_kw=dict(hspace=0)
    )
    axes[0].set_ylabel("$v_θ/v_a$")
    axes[1].set_ylabel("$v_θ/v_f$")

    colors = distinct_color_map(num_solutions)

    for id_num, (soln, color) in enumerate(zip(solutions, colors)):
        solution = soln.solution
        angles = soln.angles

        wave_speeds = sqrt(mhd_wave_speeds(
            solution[:, MAGNETIC_INDEXES], solution[:, ODEIndex.ρ], 1
        ))
        alfven = wave_speeds[MHD_Wave_Index.alfven]
        fast = wave_speeds[MHD_Wave_Index.fast]

        indexes = (start <= degrees(angles)) & (degrees(angles) <= stop)

        axes[0].plot(
            degrees(angles[indexes]),
            solution[indexes, ODEIndex.v_θ] / alfven[indexes],
            linestyle, color=color, label=str(id_num),
        )
        axes[1].plot(
            degrees(angles[indexes]),
            solution[indexes, ODEIndex.v_θ] / fast[indexes],
            linestyle, color=color, label=str(id_num)
        )

    for ax in axes:
        ax.legend(loc=0)
        ax.set_xlabel("angle from plane (°)")

    return fig
