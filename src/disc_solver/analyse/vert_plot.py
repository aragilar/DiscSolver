# -*- coding: utf-8 -*-
"""
Plot command for DiscSolver
"""
from numpy import degrees, sqrt, ones as np_ones

from ..utils import (
    mhd_wave_speeds, MHD_Wave_Index, CylindricalODEIndex,
    VERT_MAGNETIC_INDEXES, convert_spherical_to_cylindrical,
    get_vertical_scaling,
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
    stop=90, figargs=None, title=None, close=True, filename,
    mpl_style=DEFAULT_MPL_STYLE, with_version=True
):
    """
    Plot solution to file
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_plot(
        soln, soln_range, linestyle=linestyle, with_slow=with_slow,
        with_alfven=with_alfven, with_fast=with_fast, with_sonic=with_sonic,
        stop=stop, figargs=figargs, title=title, filename=filename,
        mpl_style=mpl_style, with_version=with_version,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@single_solution_plotter
def generate_plot(
    fig, soln, *, linestyle='-', with_slow=False, with_alfven=False,
    with_fast=False, with_sonic=False, stop=90, use_E_r=False
):
    """
    Generate plot, with enough freedom to be able to format fig
    """
    solution = soln.solution
    angles = soln.angles
    inp = soln.solution_input
    cons = soln.initial_conditions

    heights, vert_soln = convert_spherical_to_cylindrical(
        angles, solution, γ=inp.γ, c_s_on_v_k=inp.c_s_on_v_k, use_E_r=use_E_r,
    )

    wave_speeds = sqrt(mhd_wave_speeds(
        vert_soln[:, VERT_MAGNETIC_INDEXES],
        vert_soln[:, CylindricalODEIndex.ρ], 1,
        index=CylindricalODEIndex.B_z,
    ))

    indexes = heights <= stop

    ordering = PlotOrdering.E_r_vert if use_E_r else (
        PlotOrdering.B_φ_prime_vert
    )

    param_names = get_common_arguments(
        ordering, initial_conditions=cons, v_φ_offset=sqrt(
            cons.norm_kepler_sq / get_vertical_scaling(
                angles, c_s_on_v_k=inp.c_s_on_v_k
            )
        ),
    )

    if with_slow:
        param_names[CylindricalODEIndex.v_z]["extras"].append({
            "label": "slow",
            "data": wave_speeds[MHD_Wave_Index.slow],
        })
    if with_alfven:
        param_names[CylindricalODEIndex.v_z]["extras"].append({
            "label": "alfven",
            "data": wave_speeds[MHD_Wave_Index.alfven],
        })
    if with_fast:
        param_names[CylindricalODEIndex.v_z]["extras"].append({
            "label": "fast",
            "data": wave_speeds[MHD_Wave_Index.fast],
        })
    if with_sonic:
        param_names[CylindricalODEIndex.v_z]["extras"].append({
            "label": "sound",
            "data": np_ones(len(solution)),
        })

    axes = fig.subplots(
        nrows=2, ncols=4, sharex=True, gridspec_kw=dict(hspace=0),
    )

    # only add label to bottom plots
    for ax in axes[1]:
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
        ax.set_ylabel(settings["name"])
        ax.set_yscale(settings.get("scale", "linear"))
        if settings.get("legend"):
            ax.legend(loc=0)
    return fig
