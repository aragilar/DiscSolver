# -*- coding: utf-8 -*-
"""
Combine-plot command for DiscSolver
"""
from collections import OrderedDict

import numpy as np
from numpy import degrees, sqrt

from ..constants import KM
from ..utils import (
    mhd_wave_speeds, MHD_Wave_Index, get_normalisation, ODEIndex,
    MAGNETIC_INDEXES
)

from .utils import (
    single_solution_plotter, analyse_main_wrapper, analysis_func_wrapper,
    common_plotting_options, get_common_plot_args, plot_output_wrapper,
    DEFAULT_MPL_STYLE,
)


def plot_parser(parser):
    """
    Add arguments for plot command to parser
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
        "with_alfven": args.get("with_alfven", False),
        "with_fast": args.get("with_fast", False),
        "with_sonic": args.get("with_sonic", False),
    }


@analyse_main_wrapper(
    "Nice plotter for DiscSolver",
    plot_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
        "plot_args": get_plot_args,
    }
)
def combine_main(soln, *, soln_range, common_plot_args, plot_args):
    """
    Entry point for ds-combine-plot
    """
    return combine_plot(
        soln, soln_range=soln_range, **common_plot_args, **plot_args
    )


@analysis_func_wrapper
def combine_plot(
    soln, *, soln_range=None, plot_filename=None, show=False, linestyle='-',
    with_slow=False, with_alfven=False, with_fast=False, with_sonic=False,
    stop=90, figargs=None, title=None, close=True, filename,
    mpl_style=DEFAULT_MPL_STYLE
):
    """
    Plot solution to file, with velocities, fields onto on one plot
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_plot_combine(
        soln, soln_range, linestyle=linestyle, with_slow=with_slow,
        with_alfven=with_alfven, with_fast=with_fast, with_sonic=with_sonic,
        stop=stop, figargs=figargs, title=title, filename=filename,
        mpl_style=mpl_style,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@single_solution_plotter
def generate_plot_combine(
    fig, soln, *, linestyle='-', with_slow=False, with_alfven=False,
    with_fast=False, with_sonic=False, stop=90, use_E_r=False
):
    """
    Generate plot, with enough freedom to be able to format fig.
    Combine velocities, fields onto on plot
    """
    # pylint: disable=unused-argument
    solution = soln.solution
    angles = soln.angles
    cons = soln.initial_conditions
    inp = soln.solution_input

    norms = get_normalisation(inp)  # need to allow config here
    B_norm, v_norm, ρ_norm = norms["B_norm"], norms["v_norm"], norms["ρ_norm"]
    wave_speeds = np.sqrt(mhd_wave_speeds(
        solution[:, MAGNETIC_INDEXES], solution[:, ODEIndex.ρ], 1
    ))

    plot_props = OrderedDict([
        ("velocity", {
            "normalisation": v_norm / KM,  # km/s
            "y_label": "Velocity Field (km/s)",
            "lines": [
                {
                    "label": "$v_r$",
                    "index": ODEIndex.v_r,
                },
                {
                    "label": "$v_φ$",
                    "index": ODEIndex.v_φ,
                    "offset": sqrt(cons.norm_kepler_sq) * v_norm / KM
                },
                {
                    "label": "$v_θ$",
                    "index": ODEIndex.v_θ,
                },
            ],
        }),
        ("density", {
            "y_label": "Density (g cm$^{-3}$)",
            "normalisation": ρ_norm,
            "scale": "log",
            "lines": [
                {
                    "label": "$ρ$",
                    "index": ODEIndex.ρ,
                }
            ],
        }),
        ("fields", {
            "normalisation": B_norm,
            "y_label": "Magnetic Field (G)",
            "lines": [
                {
                    "label": "$B_r$",
                    "index": ODEIndex.B_r,
                },
                {
                    "label": "$B_φ$",
                    "index": ODEIndex.B_φ,
                },
                {
                    "label": "$B_θ$",
                    "index": ODEIndex.B_θ,
                },
            ],
        }),
    ])

    if with_slow:
        plot_props["velocity"]["lines"].append({
            "label": "slow",
            "data": wave_speeds[MHD_Wave_Index.slow],
        })
    if with_alfven:
        plot_props["velocity"]["lines"].append({
            "label": "alfven",
            "data": wave_speeds[MHD_Wave_Index.alfven],
        })
    if with_fast:
        plot_props["velocity"]["lines"].append({
            "label": "fast",
            "data": wave_speeds[MHD_Wave_Index.fast],
        })
    if with_sonic:
        plot_props["velocity"]["lines"].append({
            "label": "$c_s$",
            "data": np.ones(len(solution)),
        })

    indexes = degrees(angles) <= stop

    axes = fig.subplots(
        nrows=3, ncols=1, sharex=True, gridspec_kw=dict(hspace=0),
    )
    axes.shape = len(plot_props)
    for i, plot_name in enumerate(plot_props):
        ax = axes[i]
        settings = plot_props[plot_name]
        for line in settings["lines"]:
            if line.get("index") is not None:
                data = solution[:, line["index"]]
            else:
                data = line["data"]
            ax.plot(
                degrees(angles[indexes]),
                (
                    data[indexes] * settings["normalisation"] -
                    line.get("offset", 0)
                ), linestyle, label=line["label"]
            )
        if i == len(plot_props) - 1:  # label only the bottom one
            ax.set_xlabel("θ — angle from plane (°)")
        ax.set_ylabel(settings["y_label"])
        ax.set_yscale(settings.get("scale", "linear"))
        ax.legend(loc=0)
    fig.subplots_adjust(hspace=0)
    return fig
