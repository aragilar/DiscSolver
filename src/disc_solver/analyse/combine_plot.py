# -*- coding: utf-8 -*-
"""
Combine-plot command for DiscSolver
"""
from collections import OrderedDict
from math import sqrt

import numpy as np
from numpy import degrees
import matplotlib.pyplot as plt

from ..constants import KM
from ..utils import (
    better_sci_format, mhd_wave_speeds, MHD_WAVE_INDEX, get_normalisation,
    ODEIndex, MAGNETIC_INDEXES
)

from .utils import (
    single_solution_plotter, analyse_main_wrapper, analysis_func_wrapper,
    common_plotting_options, get_common_plot_args, savefig
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
    stop=90, figargs=None, title=None
):
    """
    Plot solution to file, with velocities, fields onto on one plot
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_plot_combine(
        soln, soln_range, linestyle=linestyle, with_slow=with_slow,
        with_alfven=with_alfven, with_fast=with_fast, with_sonic=with_sonic,
        stop=stop, figargs=figargs, title=title,
    )

    if plot_filename is not None:
        savefig(fig, plot_filename)
    if show:
        plt.show()


@single_solution_plotter
def generate_plot_combine(
    soln, *, linestyle='-', with_slow=False, with_alfven=False,
    with_fast=False, with_sonic=False, stop=90, figargs=None
):
    """
    Generate plot, with enough freedom to be able to format fig.
    Combine velocities, fields onto on plot
    """
    if figargs is None:
        figargs = {}

    solution = soln.solution
    angles = soln.angles
    cons = soln.initial_conditions
    inp = soln.solution_input

    norms = get_normalisation(inp)  # need to allow config here
    B_norm, v_norm, ρ_norm = norms["B_norm"], norms["v_norm"], norms["ρ_norm"]
    zero_soln = np.zeros(len(solution))
    v = np.array([zero_soln, zero_soln, solution[:, ODEIndex.v_θ]])
    wave_speeds = np.sqrt(mhd_wave_speeds(
        v.T, solution[:, MAGNETIC_INDEXES], solution[:, ODEIndex.ρ],
        cons.c_s * v_norm
    ))

    plot_props = OrderedDict([
        ("velocity", {
            "normalisation": v_norm / KM,  # km/s
            "y_label": "Velocity Field (km/s)",
            "lines": [
                {
                    "label": "v_r",
                    "index": ODEIndex.v_r,
                },
                {
                    "label": "v_φ",
                    "index": ODEIndex.v_φ,
                    "offset": sqrt(cons.norm_kepler_sq) * v_norm / KM
                },
                {
                    "label": "v_θ",
                    "index": ODEIndex.v_θ,
                },
            ],
        }),
        ("density", {
            "y_label": "Density ($g cm^{-3}$)",
            "normalisation": ρ_norm,
            "scale": "log",
            "lines": [
                {
                    "label": "ρ",
                    "index": ODEIndex.ρ,
                }
            ],
        }),
        ("fields", {
            "normalisation": B_norm,
            "y_label": "Magnetic Field (G)",
            "lines": [
                {
                    "label": "B_r",
                    "index": ODEIndex.B_r,
                },
                {
                    "label": "B_φ",
                    "index": ODEIndex.B_φ,
                },
                {
                    "label": "B_θ",
                    "index": ODEIndex.B_θ,
                },
            ],
        }),
    ])

    if with_slow:
        plot_props["velocity"]["lines"].append({
            "label": "slow",
            "data": wave_speeds[MHD_WAVE_INDEX["slow"]],
        })
    if with_alfven:
        plot_props["velocity"]["lines"].append({
            "label": "alfven",
            "data": wave_speeds[MHD_WAVE_INDEX["alfven"]],
        })
    if with_fast:
        plot_props["velocity"]["lines"].append({
            "label": "fast",
            "data": wave_speeds[MHD_WAVE_INDEX["fast"]],
        })
    if with_sonic:
        plot_props["velocity"]["lines"].append({
            "label": "sound",
            "data": np.ones(len(solution)),
        })

    indexes = degrees(angles) <= stop

    fig, axes = plt.subplots(
        nrows=3, ncols=1, tight_layout=True, sharex=True,
        gridspec_kw=dict(hspace=0),
        **figargs
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
            ax.set_xlabel("angle from plane (°)")
        if i % 2 == 1:
            ax.tick_params(
                axis='y', which='both', labelleft='off', labelright='on'
            )
        ax.set_ylabel(settings["y_label"])
        ax.set_yscale(settings.get("scale", "linear"))
        ax.legend(loc=0)
        better_sci_format(ax.yaxis)
    fig.subplots_adjust(hspace=0)
    return fig