# -*- coding: utf-8 -*-
"""
Diverge-plot command for DiscSolver
"""
import numpy as np
from numpy import degrees
import matplotlib.pyplot as plt
from matplotlib.style import context as use_style

from .utils import (
    plot_output_wrapper, common_plotting_options, get_common_plot_args,
    distinct_color_map, analyse_multisolution_wrapper, DEFAULT_MPL_STYLE,
)
from ..utils import (
    mhd_wave_speeds, MHD_Wave_Index, ODEIndex, MAGNETIC_INDEXES
)

NUM_ITEMS_PER_LEGEND_COLUMN = 15
SLOW_STYLE = '--'


def diverge_options(parser):
    """
    Add arguments for diverge command to parser
    """
    common_plotting_options(parser)
    parser.add_argument(
        "--with-slow", action='store_true', default=False)
    parser.add_argument(
        "--with-alfven", action='store_true', default=False)
    parser.add_argument(
        "--with-fast", action='store_true', default=False)
    parser.add_argument(
        "--with-sonic", action='store_true', default=False)
    return parser


def get_plot_args(args):
    """
    Parse plot args
    """
    return {
        "with_slow": args.get("with_slow", False),
    }


@analyse_multisolution_wrapper(
    "Show divergence in v_θ of solutions",
    diverge_options,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
        "plot_args": get_plot_args,
    }
)
def diverge_main(*solutions, common_plot_args, plot_args):
    """
    Entry point for ds-diverge-plot
    """
    return diverge_plot(*solutions, **plot_args, **common_plot_args)


def generate_diverge_plot(
    *solutions, figargs=None, stop=90, linestyle='-', with_slow=False
):
    """
    Generate plot to compare how different runs change in v_θ
    """
    if figargs is None:
        figargs = {}
    fig, ax = plt.subplots(**figargs)

    colors = distinct_color_map(len(solutions))

    num_lines = 0
    initial_plot = True

    for i, (soln, color) in enumerate(zip(solutions, colors)):
        solution = soln.solution
        angles = soln.angles
        indexes = degrees(angles) <= stop
        num_lines += 1
        ax.plot(
            degrees(angles[indexes]), solution[indexes, ODEIndex.v_θ],
            label=str(i), color=color, linestyle=linestyle,
        )

        wave_speeds = np.sqrt(mhd_wave_speeds(
            solution[:, MAGNETIC_INDEXES], solution[:, ODEIndex.ρ], 1.0
        ))

        if initial_plot:
            initial_plot = False
            initial_plot = False
            if with_slow:
                ax.plot(
                    degrees(angles[indexes]),
                    wave_speeds[MHD_Wave_Index.slow][indexes],
                    label="slow", color=color, linestyle=SLOW_STYLE,
                )
                num_lines += 1
    ax.set_xlabel("θ — angle from plane (°)")
    ax.set_ylabel("$v_θ / c_s$")
    ax.legend(ncol=max(1, num_lines//NUM_ITEMS_PER_LEGEND_COLUMN))
    return fig


def diverge_plot(
    *solutions, plot_filename=None, show=False, stop=90, figargs=None,
    title=None, linestyle='-', with_slow=False, close=True,
    mpl_style=DEFAULT_MPL_STYLE
):
    """
    Plot solution to file, with velocities, fields onto on one plot
    """
    with use_style(mpl_style):
        fig = generate_diverge_plot(
            *solutions, stop=stop, figargs=figargs, linestyle=linestyle,
            with_slow=with_slow,
        )
        if title is None:
            fig.suptitle("Comparison of v_θ")
        else:
            fig.suptitle(title)

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )
