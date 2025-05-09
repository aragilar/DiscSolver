# -*- coding: utf-8 -*-
"""
Plot command for DiscSolver
"""
from numpy import degrees, sqrt, ones as np_ones

from ..critical_points import mhd_wave_speeds
from ..utils import (
    MHD_Wave_Index, ODEIndex,
)

from .utils import (
    single_solution_plotter, analyse_main_wrapper, analysis_func_wrapper,
    common_plotting_options, get_common_plot_args, plot_output_wrapper,
    DEFAULT_MPL_STYLE, get_common_arguments, PlotOrdering,
)


def plot_parser(parser):
    """
    Add arguments for plot command to parser
    """
    common_plotting_options(parser)
    parser.add_argument("--v_θ", choices=("log", "linear"), default="linear")
    parser.add_argument("--with-slow", action='store_true', default=False)
    parser.add_argument("--with-alfven", action='store_true', default=False)
    parser.add_argument("--with-fast", action='store_true', default=False)
    parser.add_argument("--with-sonic", action='store_true', default=False)
    parser.add_argument("--portrait", action='store_true', default=False)
    parser.add_argument("--hide-roots", action='store_true', default=False)
    parser.add_argument(
        "--hide-sonic-angle", action='store_true', default=False
    )
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
        "portrait": args.get("portrait", False),
        "hide_roots": args.get("hide_roots", False),
        "hide_sonic_angle": args.get("hide_sonic_angle", False),
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
    run, *, soln_range=None, plot_filename=None, show=False, linestyle='-',
    with_slow=False, with_alfven=False, with_fast=False, with_sonic=False,
    start=0, stop=90, figargs=None, v_θ_scale="linear", title=None, close=True,
    filename, mpl_style=DEFAULT_MPL_STYLE, with_version=True, portrait=False,
    hide_roots=False, hide_sonic_angle=False,
):
    """
    Plot solution to file
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_plot(
        run, soln_range, linestyle=linestyle, with_slow=with_slow,
        with_alfven=with_alfven, with_fast=with_fast, with_sonic=with_sonic,
        start=start, stop=stop, figargs=figargs, v_θ_scale=v_θ_scale,
        title=title, filename=filename, mpl_style=mpl_style,
        with_version=with_version, portrait=portrait, hide_roots=hide_roots,
        hide_sonic_angle=hide_sonic_angle,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@single_solution_plotter
def generate_plot(
    fig, soln, *, linestyle='-', with_slow=False, with_alfven=False,
    with_fast=False, with_sonic=False, start=0, stop=90, v_θ_scale="linear",
    use_E_r=False, portrait=False, hide_roots=False, hide_sonic_angle=False,
):
    """
    Generate plot, with enough freedom to be able to format fig
    """
    nrows = 4 if portrait else 2
    ncols = 2 if portrait else 4

    solution = soln.solution
    angles = soln.angles
    cons = soln.initial_conditions
    sonic_angle = soln.sonic_point
    sonic_values = soln.sonic_point_values
    roots_angles = soln.t_roots
    roots_values = soln.y_roots

    wave_speeds = sqrt(mhd_wave_speeds(values=solution))

    indexes = (start <= degrees(angles)) & (degrees(angles) <= stop)

    ordering = PlotOrdering.E_r if use_E_r else PlotOrdering.B_φ_prime

    param_names = get_common_arguments(
        ordering, v_θ_scale=v_θ_scale, initial_conditions=cons
    )

    if with_slow:
        param_names[ODEIndex.v_θ]["extras"].append({
            "label": "slow",
            "data": wave_speeds[MHD_Wave_Index.slow],
        })
    if with_alfven:
        param_names[ODEIndex.v_θ]["extras"].append({
            "label": "alfven",
            "data": wave_speeds[MHD_Wave_Index.alfven],
        })
    if with_fast:
        param_names[ODEIndex.v_θ]["extras"].append({
            "label": "fast",
            "data": wave_speeds[MHD_Wave_Index.fast],
        })
    if with_sonic:
        param_names[ODEIndex.v_θ]["extras"].append({
            "label": "sound",
            "data": np_ones(len(solution)),
        })

    axes = fig.subplots(
        nrows=nrows, ncols=ncols, sharex=True, gridspec_kw=dict(hspace=0),
    )

    # only add label to bottom plots
    for ax in axes[-1]:
        ax.set_xlabel("angle from plane (°)")

    axes.shape = len(param_names)
    for i, settings in enumerate(param_names):
        ax = axes[i]
        ax.plot(
            degrees(angles[indexes]),
            (
                solution[:, i] - settings.get("offset", 0)
            )[indexes], linestyle, label=settings["name"]
        )
        for extra in settings.get("extras", []):
            ax.plot(
                degrees(angles[indexes]),
                extra["data"][indexes],
                label=extra.get("label")
            )
        if sonic_angle is not None and not hide_sonic_angle:
            ax.plot(
                degrees(sonic_angle),
                sonic_values[i] - settings.get("offset", 0),
                linestyle=None, marker='.', color='red',
            )
        if roots_angles is not None and not hide_roots:
            ax.plot(
                degrees(roots_angles),
                roots_values[:, i] - settings.get("offset", 0),
                linestyle=None, marker='.', color='black',
            )
        ax.set_ylabel(settings["name"])
        ax.set_yscale(settings.get("scale", "linear"))
        if settings.get("legend"):
            ax.legend(loc=0)
    return fig
