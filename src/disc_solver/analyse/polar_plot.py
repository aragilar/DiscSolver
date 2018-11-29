# -*- coding: utf-8 -*-
"""
Plot command for DiscSolver
"""
from numpy import degrees, sqrt, ones as np_ones, logspace, nan, full, linspace

from ..utils import (
    mhd_wave_speeds, MHD_Wave_Index, ODEIndex, MAGNETIC_INDEXES,
    scale_solution_to_radii,
)

from .utils import (
    single_solution_plotter, analyse_main_wrapper, analysis_func_wrapper,
    common_plotting_options, get_common_plot_args, plot_output_wrapper,
    DEFAULT_MPL_STYLE, get_common_arguments, B_φ_PRIME_ORDERING, E_r_ORDERING,
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
    stop=90, figargs=None, v_θ_scale="linear", title=None, close=True,
    filename, mpl_style=DEFAULT_MPL_STYLE, with_version=True
):
    """
    Plot solution to file
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_plot(
        soln, soln_range, linestyle=linestyle, with_slow=with_slow,
        with_alfven=with_alfven, with_fast=with_fast, with_sonic=with_sonic,
        stop=stop, figargs=figargs, v_θ_scale=v_θ_scale, title=title,
        filename=filename, mpl_style=mpl_style, with_version=with_version,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@single_solution_plotter
def generate_plot(
    fig, soln, *, linestyle='-', with_slow=False, with_alfven=False,
    with_fast=False, with_sonic=False, stop=90, v_θ_scale="linear",
    use_E_r=False
):
    """
    Generate plot, with enough freedom to be able to format fig
    """
    solution = soln.solution
    angles = soln.angles
    cons = soln.initial_conditions
    sonic_angle = soln.sonic_point
    sonic_values = soln.sonic_point_values
    roots_angles = soln.t_roots
    roots_values = soln.y_roots

    γ = cons.γ

    wave_speeds = sqrt(mhd_wave_speeds(
        solution[:, MAGNETIC_INDEXES], solution[:, ODEIndex.ρ], 1
    ))

    indexes = degrees(angles) <= stop

    ordering = E_r_ORDERING if use_E_r else B_φ_PRIME_ORDERING

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

    r_values = linspace(0.1, 10)
    print(r_values)
    print(degrees(angles))
    print(angles)

    all_mapped_vals = full([r_values.shape[0]] + list(solution.shape), nan)

    for i, r in enumerate(r_values):
        all_mapped_vals[i] = scale_solution_to_radii(
            solution, r, γ=γ, use_E_r=use_E_r,
        )

    v_r = all_mapped_vals[:, indexes, ODEIndex.v_r]
    v_θ = all_mapped_vals[:, indexes, ODEIndex.v_θ]

    #axes = fig.subplots(
    #    nrows=2, ncols=3, gridspec_kw=dict(hspace=0),
    #    subplot_kw=dict(projection="polar"),
    #)
    #v_ax = axes[0,0]
    v_ax = fig.subplots(
        gridspec_kw=dict(hspace=0),
        subplot_kw=dict(projection="polar"),
    )

    v_ax.set_thetamin(0)
    v_ax.set_thetamax(stop)
    import pdb; pdb.set_trace()
    v_ax.streamplot(angles[indexes], r_values, v_θ, v_r)
    return fig
