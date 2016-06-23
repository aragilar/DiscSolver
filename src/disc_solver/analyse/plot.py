# -*- coding: utf-8 -*-
"""
Plot command for DiscSolver
"""
from math import sqrt

from numpy import (
    degrees, array as np_array, zeros as np_zeros, sqrt as np_sqrt,
    ones as np_ones
)
import matplotlib.pyplot as plt

from ..constants import KM
from ..utils import (
    better_sci_format, mhd_wave_speeds, MHD_WAVE_INDEX, get_normalisation,
    ODEIndex, MAGNETIC_INDEXES,
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
    parser.add_argument("--v_θ", choices=("log", "linear"), default="linear")
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
        "v_θ_scale": args.get("v_θ", "linear"),
        "with_slow": args.get("with_slow", False),
        "with_alfven": args.get("with_alfven", False),
        "with_fast": args.get("with_fast", False),
        "with_sonic": args.get("with_sonic", False),
    }


@analyse_main_wrapper(
    "Main plotter for DiscSolver",
    plot_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
        "plot_args": get_plot_args,
    }
)
def plot_main(soln, *, soln_range, common_plot_args, plot_args):
    """
    Entry point for ds-plot
    """
    return plot(soln, soln_range=soln_range, **common_plot_args, **plot_args)


@analysis_func_wrapper
def plot(
    soln, *, soln_range=None, plot_filename=None, show=False, linestyle='-',
    with_slow=False, with_alfven=False, with_fast=False, with_sonic=False,
    stop=90, figargs=None, v_θ_scale="linear"
):
    """
    Plot solution to file
    """
    # pylint: disable=too-many-function-args
    fig = generate_plot(
        soln, soln_range, linestyle=linestyle, with_slow=with_slow,
        with_alfven=with_alfven, with_fast=with_fast, with_sonic=with_sonic,
        stop=stop, figargs=figargs, v_θ_scale=v_θ_scale,
    )

    if plot_filename is not None:
        savefig(fig, plot_filename)
    if show:
        plt.show()


@single_solution_plotter
def generate_plot(
    soln, *, linestyle='-', with_slow=False, with_alfven=False,
    with_fast=False, with_sonic=False, stop=90, figargs=None,
    v_θ_scale="linear"
):
    """
    Generate plot, with enough freedom to be able to format fig
    """
    if figargs is None:
        figargs = {}

    solution = soln.solution
    angles = soln.angles
    cons = soln.initial_conditions
    inp = soln.solution_input
    y_roots = soln.y_roots
    t_roots = soln.t_roots

    norms = get_normalisation(inp)  # need to allow config here
    B_norm, v_norm, ρ_norm = norms["B_norm"], norms["v_norm"], norms["ρ_norm"]
    zero_soln = np_zeros(len(solution))
    v = np_array([zero_soln, zero_soln, solution[:, ODEIndex.v_θ]])
    wave_speeds = np_sqrt(mhd_wave_speeds(
        v.T, solution[:, MAGNETIC_INDEXES], solution[:, 6], cons.c_s * v_norm
    ))

    indexes = degrees(angles) <= stop

    param_names = [
        {
            "name": "B_r",
            "y_label": "Magnetic Field (Gauss)",
            "normalisation": B_norm,
        },
        {
            "name": "B_φ",
            "y_label": "Magnetic Field (Gauss)",
            "normalisation": B_norm,
        },
        {
            "name": "B_θ",
            "y_label": "Magnetic Field (Gauss)",
            "normalisation": B_norm,
        },
        {
            "name": "v_r",
            "y_label": "Velocity Field (km/s)",
            "normalisation": v_norm / KM,  # km/s
        },
        {
            "name": "v_φ",
            "y_label": "Velocity Field (km/s)",
            "normalisation": v_norm / KM,  # km/s
            "offset": sqrt(cons.norm_kepler_sq) * v_norm / KM
        },
        {
            "name": "v_θ",
            "y_label": "Velocity Field (km/s)",
            "normalisation": v_norm / KM,  # km/s
            "legend": True,
            "scale": v_θ_scale,
            "extras": []
        },
        {
            "name": "ρ",
            "y_label": "Density ($g cm^{-3}$)",
            "normalisation": ρ_norm,
            "scale": "log",
        },
        {
            "name": "B_φ_prime",
            "y_label": "Magnetic Field (Gauss)",
            "normalisation": B_norm,
        },
    ]

    if with_slow:
        param_names[ODEIndex.v_θ]["extras"].append({
            "label": "slow",
            "data": wave_speeds[MHD_WAVE_INDEX["slow"]],
            "normalisation": v_norm / KM,
        })
    if with_alfven:
        param_names[ODEIndex.v_θ]["extras"].append({
            "label": "alfven",
            "data": wave_speeds[MHD_WAVE_INDEX["alfven"]],
            "normalisation": v_norm / KM,
        })
    if with_fast:
        param_names[ODEIndex.v_θ]["extras"].append({
            "label": "fast",
            "data": wave_speeds[MHD_WAVE_INDEX["fast"]],
            "normalisation": v_norm / KM,
        })
    if with_sonic:
        param_names[ODEIndex.v_θ]["extras"].append({
            "label": "sound",
            "data": np_ones(len(solution)),
            "normalisation": v_norm / KM,
        })

    fig, axes = plt.subplots(
        nrows=2, ncols=4, tight_layout=True, sharex=True, **figargs
    )
    axes.shape = len(param_names)
    for i, settings in enumerate(param_names):
        ax = axes[i]
        ax.plot(
            degrees(angles[indexes]),
            (
                solution[:, i] * settings["normalisation"] -
                settings.get("offset", 0)
            )[indexes], linestyle,
        )
        for extra in settings.get("extras", []):
            ax.plot(
                degrees(angles[indexes]),
                (extra["data"] * extra["normalisation"])[indexes],
                label=extra.get("label")
            )
        ax.set_xlabel("angle from plane (°)")
        ax.set_ylabel(settings["y_label"])
        ax.set_yscale(settings.get("scale", "linear"))
        ax.set_title(settings["name"])
        if settings.get("legend"):
            ax.legend(loc=0)
        better_sci_format(ax.yaxis)
        if t_roots is not None:
            ax.axvline(degrees(t_roots[0]))
            ax.plot(degrees(t_roots[0]), y_roots[0, i], ".")
    return fig
