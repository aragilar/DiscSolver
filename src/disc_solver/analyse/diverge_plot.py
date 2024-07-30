# -*- coding: utf-8 -*-
"""
Diverge-plot command for DiscSolver
"""
import logbook
import numpy as np
from numpy import degrees
import matplotlib.pyplot as plt
from matplotlib.style import context as use_style

from .utils import (
    plot_output_wrapper, common_plotting_options, get_common_plot_args,
    distinct_color_map, analyse_multisolution_wrapper, DEFAULT_MPL_STYLE,
    add_version_to_plot, add_label_display_on_select,
)
from ..utils import (
    mhd_wave_speeds, MHD_Wave_Index, ODEIndex, MAGNETIC_INDEXES
)

logger = logbook.Logger(__name__)

NUM_ITEMS_PER_LEGEND_COLUMN = 15
SLOW_STYLE = '-.'
ALFVEN_STYLE = '--'
FAST_STYLE = '-.'


def diverge_options(parser):
    """
    Add arguments for diverge command to parser
    """
    common_plotting_options(parser)
    parser.add_argument("--with-slow", action='store_true', default=False)
    parser.add_argument("--with-alfven", action='store_true', default=False)
    parser.add_argument("--with-fast", action='store_true', default=False)
    parser.add_argument("--solutions", action='store', default=None)
    parser.add_argument(
        "--hide-legend", action='store_false', default=True,
        dest="show_legend",
    )
    return parser


def parse_solution_filter(filter_string):
    """
    Parse solution filter yielding solution strings.
    """
    for entry in filter_string.strip().split(','):
        entry = entry.strip()
        if '-' in entry:
            start, stop = entry.split('-', 2)
            for i in range(int(start), int(stop)):
                yield str(i)
            yield stop
        else:
            yield entry


def get_plot_args(args):
    """
    Parse plot args
    """
    solution_filter = args.get("solutions", None)
    if solution_filter is not None:
        solution_filter = list(parse_solution_filter(solution_filter))
    return {
        "with_slow": args.get("with_slow", False),
        "with_alfven": args.get("with_alfven", False),
        "with_fast": args.get("with_fast", False),
        "show_legend": args.get("show_legend", True),
        "solution_filter": solution_filter,
    }


@analyse_multisolution_wrapper(
    "Show divergence in v_θ of solutions",
    diverge_options,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
        "plot_args": get_plot_args,
    }
)
def diverge_main(*, solutions, common_plot_args, plot_args, filename):
    """
    Entry point for ds-diverge-plot
    """
    return diverge_plot(
        solutions, **plot_args, **common_plot_args, filename=filename
    )


def generate_diverge_plot(
    solutions, *, figargs=None, start=0, stop=90, linestyle='-',
    with_slow=False, with_alfven=False, with_fast=False, show_legend=True,
):
    """
    Generate plot to compare how different runs change in v_θ
    """
    if figargs is None:
        figargs = {}
    fig, ax = plt.subplots(constrained_layout=True, **figargs)

    colors = distinct_color_map(len(solutions))

    num_lines = 0

    for (soln_name, soln), color in zip(solutions, colors):
        solution = soln.solution
        angles = soln.angles
        indexes = (start <= degrees(angles)) & (degrees(angles) <= stop)
        num_lines += 1
        ax.plot(
            degrees(angles[indexes]), solution[indexes, ODEIndex.v_θ],
            label=soln_name, color=color, linestyle=linestyle,
        )

        wave_speeds = np.sqrt(mhd_wave_speeds(
            solution[:, MAGNETIC_INDEXES], solution[:, ODEIndex.ρ], 1.0
        ))

        if with_slow:
            ax.plot(
                degrees(angles[indexes]),
                wave_speeds[MHD_Wave_Index.slow][indexes],
                label="slow", color=color, linestyle=SLOW_STYLE,
            )
            num_lines += 1

        if with_alfven:
            ax.plot(
                degrees(angles[indexes]),
                wave_speeds[MHD_Wave_Index.alfven][indexes],
                label="alfven", color=color, linestyle=ALFVEN_STYLE,
            )
            num_lines += 1

        if with_fast:
            ax.plot(
                degrees(angles[indexes]),
                wave_speeds[MHD_Wave_Index.fast][indexes],
                label="fast", color=color, linestyle=FAST_STYLE,
            )
            num_lines += 1

    ax.set_xlabel("$θ$ — angle from plane (°)")
    ax.set_ylabel("$v_θ / c_s$")
    if show_legend:
        ax.legend(ncol=max(1, num_lines//NUM_ITEMS_PER_LEGEND_COLUMN))
    add_label_display_on_select(ax)
    return fig


def diverge_plot(
    solutions, *, plot_filename=None, show=False, start=0, stop=90,
    figargs=None, title=None, linestyle='-', with_slow=False, close=True,
    mpl_style=DEFAULT_MPL_STYLE, with_version=True, with_alfven=False,
    with_fast=False, filename, show_legend=True, solution_filter=None,
):
    """
    Plot solution to file, with velocities, fields onto on one plot
    """
    if solution_filter is not None:
        soln_map = dict(solutions)
        solutions = [
            (soln_id, soln_map[soln_id]) for soln_id in solution_filter
        ]
    with use_style(mpl_style):
        fig = generate_diverge_plot(
            solutions, start=start, stop=stop, figargs=figargs,
            linestyle=linestyle, with_slow=with_slow, with_alfven=with_alfven,
            with_fast=with_fast, show_legend=show_legend,
        )
        if title is None:
            fig.suptitle(f"Comparison of v_θ for {filename}")
        elif title:
            fig.suptitle(title)

    if with_version:
        add_version_to_plot(fig)

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )
